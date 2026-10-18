"""Check the UMD download once; launch the authorized baseline at most once."""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tarfile
import time
import traceback

from ..common import read, write, sha


TERMINAL = {"complete", "complete_with_failures", "failed"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def safe_member(member, root_name):
    path = PurePosixPath(member.name)
    if (path.is_absolute() or not path.parts or path.parts[0] != root_name
            or ".." in path.parts or "\\" in member.name
            or not (member.isfile() or member.isdir())):
        raise ValueError(f"Unsupported archive member: {member.name}")
    return path


def inspect_archive(archive, root_name):
    """Read through the gzip trailer (CRC), and validate every tar entry."""
    before = fingerprint(archive)
    count = total = allocated = 0
    names = set()
    with gzip.open(archive, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as bundle:
            for member in bundle:
                name = str(safe_member(member, root_name))
                if name in names:
                    raise ValueError(f"Duplicate archive member: {name}")
                names.add(name)
                count += 1
                total += member.size
                allocated += max(4096, (member.size + 4095) // 4096 * 4096)
        # Tar iteration can stop before the gzip trailer. Explicitly verify it.
        while stream.read(8 * 1024**2):
            pass
    for required in ["novel_split.txt", "category_10_fold.txt", "README.txt", "tools"]:
        if f"{root_name}/{required}" not in names:
            raise ValueError(f"Archive missing {required}")
    digest = sha(archive)
    if before != fingerprint(archive):
        raise RuntimeError("Archive changed while being verified")
    return {"checked_utc": utc(), "fingerprint": before, "sha256": digest,
            "gzip_crc_valid": True, "tar_members": count,
            "uncompressed_bytes": total, "estimated_extracted_bytes": allocated}


def extract_archive(archive, destination, inventory):
    """Stage extraction and publish the dataset only after every file is written."""
    destination = Path(destination)
    marker = destination / ".umd_archive.json"
    if destination.exists():
        if marker.is_file() and read(marker)["sha256"] == inventory["sha256"]:
            return
        raise FileExistsError(f"Existing dataset is not verified by this task: {destination}")
    staging = destination.parent / ("." + destination.name + ".extracting")
    staging.mkdir()  # Never overwrite a previous partial extraction.
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            relative = safe_member(member, destination.name)
            path = staging.joinpath(*relative.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024**2)
    if fingerprint(archive) != inventory["fingerprint"]:
        raise RuntimeError("Archive changed during extraction")
    write(staging / destination.name / ".umd_archive.json", inventory)
    (staging / destination.name).rename(destination)
    staging.rmdir()


def download_processes(archive):
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            command = Path(os.fsdecode(args[0])).name
            if command not in {"wget", "curl", "aria2c"}:
                continue
            if any(Path(archive).name.encode() in arg for arg in args[1:]):
                matches.append({"pid": int(entry.name), "command": command})
        except (OSError, IndexError):
            pass
    return matches


class Monitor:
    def __init__(self, config_path, notify=True):
        self.config_path = Path(config_path).resolve()
        self.config = read(config_path)
        self.directory = Path(self.config["monitor_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latest.json"
        self.previous = read(self.path) if self.path.exists() else {}
        self.state = dict(self.previous)
        self.notify_enabled = notify

    def record(self, status, **fields):
        self.state.update(status=status, checked_utc=utc(), **fields)
        write(self.path, self.state)
        with (self.directory / "checks.jsonl").open("a", encoding="utf-8") as stream:
            import json
            stream.write(json.dumps(self.state, ensure_ascii=False) + "\n")
        print(f"[{self.state['checked_utc']}] {status}: {fields.get('message', '')}", flush=True)

    def notify(self):
        if not self.notify_enabled:
            return
        config = self.config
        prompt = (
            "这是用户授权的 UMD 每小时下载检查/实验状态通知。"
            f"请读取 {self.path}，并核对下载进程与该目录日志，用中文简要汇报。"
            "下载完成后的校验、解压、PartGLEE＋Valen-4B 推理和评测由本地 systemd 服务自动执行；"
            "不要重复启动实验，不要修改参数或删除文件。"
            "若状态为 waiting_space，请报告空间缺口并等待用户处理；"
            "若 complete/failed，请核对结果与定时器停止状态。"
        )
        try:
            result = subprocess.run([config["codex"], "queue", "--thread", config["thread_id"],
                                     "--message", prompt], cwd=config["workspace"],
                                    capture_output=True, text=True, timeout=45)
            delivery = {"at_utc": utc(), "status": self.state["status"],
                        "returncode": result.returncode, "stdout": result.stdout.strip(),
                        "stderr": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            delivery = {"at_utc": utc(), "status": self.state["status"], "error": str(error)}
        with (self.directory / "notifications.jsonl").open("a", encoding="utf-8") as stream:
            import json
            stream.write(json.dumps(delivery, ensure_ascii=False) + "\n")
        # A chat delivery failure must not restart or abort an authorized experiment.
        print(f"Notification: {delivery}", flush=True)

    def stop_timer(self):
        result = subprocess.run(["systemctl", "--user", "disable", "--now", self.config["timer_unit"]],
                                capture_output=True, text=True, timeout=20)
        self.record(self.state["status"], timer_stopped=result.returncode == 0,
                    timer_stop_error=result.stderr.strip() if result.returncode else None)

    def command(self, stage):
        log = self.directory / f"{stage}.log"
        self.record(stage, stage_log=str(log), message=f"Running {stage}")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{utc()}] Starting {stage}\n")
            stream.flush()
            result = subprocess.run(self.config["commands"][stage], cwd=self.config["project_dir"],
                                    stdout=stream, stderr=subprocess.STDOUT, env=env)
        return result.returncode

    def check(self):
        config = self.config
        archive = Path(config["archive"])
        dataset = Path(config["dataset_dir"])
        manifest = Path(config["manifest"])
        output = Path(config["output_dir"])
        if self.state.get("status") in TERMINAL:
            return
        # The lock is held for the entire run. If it is free but this marker exists,
        # the previous launcher ended unexpectedly. Never launch a second experiment.
        if (self.directory / "launched.json").exists():
            raise RuntimeError("Previous experiment launch exists; inspect logs, do not automatically rerun")
        size = archive.stat().st_size if archive.exists() else 0
        expected = config["expected_archive_bytes"]
        delta = size - self.previous.get("downloaded_bytes", size)
        details = {"archive": str(archive), "downloaded_bytes": size, "expected_bytes": expected,
                   "progress_percent": round(100 * size / expected, 2),
                   "bytes_since_previous_check": delta, "download_processes": download_processes(archive),
                   "free_disk_bytes": shutil.disk_usage(archive.parent).free,
                   "output_dir": str(output), "timer_stopped": False}
        if size < expected:
            self.record("downloading" if details["download_processes"] else "download_incomplete",
                        **details, message="Waiting for the complete tools archive")
            return
        if size != expected:
            raise ValueError(f"Archive size {size} differs from the verified server size {expected}")
        if details["download_processes"] or time.time() - archive.stat().st_mtime < 15:
            self.record("waiting_for_writer", **details, message="Waiting for the downloader to close the archive")
            return
        cached = self.directory / "archive_verified.json"
        inventory = read(cached) if cached.exists() else {}
        if inventory.get("fingerprint") != fingerprint(archive):
            self.record("verifying", **details, message="Checking gzip CRC and tar contents")
            inventory = inspect_archive(archive, dataset.name)
            write(cached, inventory)
        reserve = config["reserve_disk_bytes"]
        if not dataset.exists():
            required = inventory["estimated_extracted_bytes"] + reserve
            free = shutil.disk_usage(dataset.parent).free
            if free < required:
                self.record("waiting_space", **details, required_free_bytes=required,
                            space_shortfall_bytes=required-free, waiting_stage="extraction",
                            message="Insufficient free space for verified extraction")
                return
        self.record("extracting", **details, archive_sha256=inventory["sha256"],
                    message="Extracting verified archive into its own staging directory")
        extract_archive(archive, dataset, inventory)
        if not manifest.exists():
            if self.command("prepare"):
                raise RuntimeError("UMD manifest preparation failed; see prepare.log")
        meta = read(manifest.with_suffix(".meta.json"))
        if (meta.get("manifest_sha256") != sha(manifest)
                or meta.get("dataset_root") != str(dataset)
                or meta.get("split") != "novel" or meta.get("partition") != "test"
                or meta.get("subset") != "tools" or not meta.get("complete_selected_split")):
            raise ValueError("Manifest is not the complete official tools novel/test manual-frame set")
        required = meta["samples"] * config["estimated_output_bytes_per_image"] + reserve
        free = shutil.disk_usage(output.parent).free
        if free < required:
            self.record("waiting_space", manual_test_images=meta["samples"], free_disk_bytes=free,
                        required_free_bytes=required, space_shortfall_bytes=required-free,
                        waiting_stage="inference", message="Insufficient space for full intermediate outputs")
            return
        gpu = subprocess.run(["nvidia-smi", f"--id={config['gpu_index']}",
                              "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=True)
        free_mib, utilization = map(int, gpu.stdout.strip().split(","))
        if free_mib < config["minimum_free_gpu_mib"] or utilization > 10:
            self.record("waiting_gpu", gpu_free_mib=free_mib, gpu_utilization=utilization,
                        message="GPU 0 is busy; will check again in one hour")
            return
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing experiment: {output}")
        # Exclusive creation, in addition to flock and systemd's single service instance.
        with (self.directory / "launched.json").open("x", encoding="utf-8") as stream:
            import json
            json.dump({"started_utc": utc(), "pid": os.getpid(), "output": str(output),
                       "manifest_sha256": sha(manifest), "samples": meta["samples"]}, stream, indent=2)
        write(self.directory / "launch_config.json", config)
        self.record("starting_experiment", manual_test_images=meta["samples"],
                    free_disk_bytes=free, gpu_free_mib=free_mib, message="Launching PartGLEE + Valen-4B once")
        self.notify()
        infer_code = self.command("infer")
        summary = read(output / "run.json")
        counts = summary.get("counts", {})
        # Individual failures are scored as background. A fatal worker exit that
        # leaves pending samples must not silently become a completed evaluation.
        if (infer_code not in {0, 1} or not counts or set(counts) - {"complete", "failed"}
                or summary.get("error") or sum(counts.values()) != meta["samples"]):
            raise RuntimeError(f"Inference did not finish all samples: exit={infer_code}, counts={counts}")
        if self.command("evaluate"):
            raise RuntimeError("UMD evaluation failed; see evaluate.log")
        metrics = output / "evaluation/metrics.json"
        if not metrics.is_file():
            raise RuntimeError("Evaluation exited without metrics.json")
        self.record("complete_with_failures" if counts.get("failed") else "complete",
                    counts=counts, metrics=str(metrics), preview=str(output / "evaluation/index.html"),
                    message="Inference and evaluation finished")

    def run(self):
        with (self.directory / "monitor.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("An existing check or experiment holds the lock", flush=True)
                return
            self.previous = read(self.path) if self.path.exists() else {}
            self.state = dict(self.previous)
            try:
                self.check()
            except BaseException as error:
                traceback.print_exc()
                self.record("failed", error=f"{type(error).__name__}: {error}",
                            message="Stopped; no automatic experiment retry")
            if self.state.get("status") in TERMINAL:
                self.stop_timer()
            self.notify()


def stopped(*_):
    raise InterruptedError("Service stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-notify", action="store_true", help="Check normally without queuing a chat report")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stopped)
    Monitor(args.config, notify=not args.no_notify).run()


if __name__ == "__main__":
    main()
