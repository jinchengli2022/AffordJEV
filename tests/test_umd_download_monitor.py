"""Check archive integrity and prevent premature/duplicate automatic launches."""
import gzip
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from affordance_jev.automation.umd_download import Monitor, inspect_archive, extract_archive
from affordance_jev.common import read, write, sha


class DownloadMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / "tools.tar.gz"
        self.root_name = "part-affordance-dataset"

    def archive_bytes(self, extra=None):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as bundle:
            for name in [self.root_name, self.root_name + "/tools"]:
                member = tarfile.TarInfo(name)
                member.type = tarfile.DIRTYPE
                bundle.addfile(member)
            for name in ["novel_split.txt", "category_10_fold.txt", "README.txt"]:
                data = b"original content\n"
                member = tarfile.TarInfo(self.root_name + "/" + name)
                member.size = len(data)
                bundle.addfile(member, io.BytesIO(data))
            if extra:
                bundle.addfile(extra)
        return gzip.compress(raw.getvalue())

    def monitor(self):
        config = {
            "archive": str(self.archive), "expected_archive_bytes": 1000,
            "dataset_dir": str(self.directory / self.root_name),
            "manifest": str(self.directory / "manifest.jsonl"),
            "output_dir": str(self.directory / "predictions"),
            "monitor_dir": str(self.directory / "monitor"),
            "reserve_disk_bytes": 0, "estimated_output_bytes_per_image": 20 * 1024**2,
        }
        path = self.directory / "config.json"
        write(path, config)
        return Monitor(path, notify=False)

    def test_crc_and_extraction_preserve_files(self):
        self.archive.write_bytes(self.archive_bytes())
        info = inspect_archive(self.archive, self.root_name)
        self.assertTrue(info["gzip_crc_valid"])
        destination = self.directory / self.root_name
        extract_archive(self.archive, destination, info)
        self.assertEqual((destination / "README.txt").read_bytes(), b"original content\n")
        self.assertEqual(read(destination / ".umd_archive.json")["sha256"], info["sha256"])
        extract_archive(self.archive, destination, info)  # Verified reuse is idempotent.

    def test_gzip_corruption_after_tar_end_is_detected(self):
        data = bytearray(self.archive_bytes())
        data[-8] ^= 1  # Tar members are intact; only the gzip CRC is corrupted.
        self.archive.write_bytes(data)
        with self.assertRaises(gzip.BadGzipFile):
            inspect_archive(self.archive, self.root_name)

    def test_unsafe_members_are_not_extracted(self):
        for member in [tarfile.TarInfo(self.root_name + "/../escape"), tarfile.TarInfo("/absolute")]:
            self.archive.write_bytes(self.archive_bytes(member))
            with self.assertRaises(ValueError):
                inspect_archive(self.archive, self.root_name)
        link = tarfile.TarInfo(self.root_name + "/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp"
        self.archive.write_bytes(self.archive_bytes(link))
        with self.assertRaises(ValueError):
            inspect_archive(self.archive, self.root_name)

    def test_partial_download_cannot_start_archive_or_models(self):
        self.archive.write_bytes(b"partial")
        monitor = self.monitor()
        with patch("affordance_jev.automation.umd_download.download_processes", return_value=[{"pid": 1}]), \
                patch("affordance_jev.automation.umd_download.inspect_archive") as inspect, \
                patch.object(monitor, "command") as command:
            monitor.check()
        self.assertEqual(monitor.state["status"], "downloading")
        inspect.assert_not_called()
        command.assert_not_called()

    def test_launch_marker_prevents_retry_after_interruption(self):
        monitor = self.monitor()
        write(monitor.directory / "launched.json", {"pid": 12345})
        with patch.object(monitor, "command") as command:
            with self.assertRaisesRegex(RuntimeError, "Previous experiment launch"):
                monitor.check()
        command.assert_not_called()

    def test_completed_run_is_not_restarted(self):
        monitor = self.monitor()
        monitor.state = {"status": "complete"}
        with patch.object(monitor, "command") as command:
            monitor.check()
        command.assert_not_called()

    def test_existing_unverified_dataset_is_preserved(self):
        self.archive.write_bytes(self.archive_bytes())
        info = inspect_archive(self.archive, self.root_name)
        destination = self.directory / self.root_name
        destination.mkdir()
        (destination / "user_file").write_text("keep")
        with self.assertRaises(FileExistsError):
            extract_archive(self.archive, destination, info)
        self.assertEqual((destination / "user_file").read_text(), "keep")

    def ready_monitor(self, failed=False):
        monitor = self.monitor()
        self.archive.write_bytes(self.archive_bytes())
        os.utime(self.archive, (100, 100))
        monitor.config.update(expected_archive_bytes=self.archive.stat().st_size,
                              gpu_index=0, minimum_free_gpu_mib=20000)

        def fake_stage(stage):
            manifest = Path(monitor.config["manifest"])
            output = Path(monitor.config["output_dir"])
            if stage == "prepare":
                manifest.write_text('{}\n')
                write(manifest.with_suffix(".meta.json"), {
                    "manifest_sha256": sha(manifest), "dataset_root": monitor.config["dataset_dir"],
                    "split": "novel", "partition": "test", "subset": "tools",
                    "complete_selected_split": True, "samples": 1})
            elif stage == "infer":
                write(output / "run.json", {"counts": {"failed" if failed else "complete": 1}, "error": None})
                return int(failed)
            elif stage == "evaluate":
                write(output / "evaluation/metrics.json", {"fixture": True})
            return 0
        return monitor, fake_stage

    def test_complete_flow_launches_once_and_evaluates_recorded_failures(self):
        monitor, stages = self.ready_monitor(failed=True)
        with patch("affordance_jev.automation.umd_download.download_processes", return_value=[]), \
                patch("affordance_jev.automation.umd_download.shutil.disk_usage", return_value=SimpleNamespace(free=1024**3)), \
                patch("affordance_jev.automation.umd_download.subprocess.run", return_value=SimpleNamespace(stdout="24000, 0")), \
                patch.object(monitor, "command", side_effect=stages) as command:
            monitor.check()
            monitor.check()
        self.assertEqual([call.args[0] for call in command.call_args_list], ["prepare", "infer", "evaluate"])
        self.assertEqual(monitor.state["status"], "complete_with_failures")
        self.assertTrue((monitor.directory / "launched.json").is_file())

    def test_output_space_shortfall_waits_without_launch_marker(self):
        monitor, stages = self.ready_monitor()
        with patch("affordance_jev.automation.umd_download.download_processes", return_value=[]), \
                patch("affordance_jev.automation.umd_download.shutil.disk_usage", return_value=SimpleNamespace(free=1024**2)), \
                patch("affordance_jev.automation.umd_download.subprocess.run") as gpu, \
                patch.object(monitor, "command", side_effect=stages) as command:
            monitor.check()
        self.assertEqual(monitor.state["status"], "waiting_space")
        self.assertEqual([call.args[0] for call in command.call_args_list], ["prepare"])
        self.assertFalse((monitor.directory / "launched.json").exists())
        gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
