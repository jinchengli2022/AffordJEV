"""Hourly read-only checks and notifications; never starts or restarts experiments."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def running(pid, expected):
    try:
        proc = Path('/proc')/str(pid)
        cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return all(x in cmd for x in expected) and (proc/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def inspect(config):
    root = Path(config['experiment'])
    state = read(root/'status.json')
    supervisor = read(root/'supervisor.json') if (root/'supervisor.json').exists() else {}
    alive = running(supervisor.get('pid', -1), [str(root), 'affordance_jev.experiments.valen_foreground.suite'])
    phase = root/state.get('phase', 'smoke')
    warnings = []
    terminal = state['status'] in ['complete', 'failed'] or (not alive and state['status'] != 'prepared')
    status = state['status']
    if terminal and status not in ['complete', 'failed']:
        status = 'failed'
        warnings.append('Supervisor exited before completion; do not restart automatically')
    progress = {p.stem: read(p) for p in sorted((phase/'progress').glob('*.json'))}
    workers = []
    for p in (phase/'processes').glob('*.json'):
        data = read(p)
        if running(data['pid'], [str(phase), 'affordance_jev.experiments.valen_foreground.workers']):
            workers.append({'worker': p.stem, 'pid': data['pid'], 'gpu': data['gpu']})
    logs = {}
    for p in [root/'pipeline.log', *sorted((phase/'logs').glob(f'{state.get("stage", "")}_*.log'))]:
        if p.is_file():
            with p.open('rb') as stream:
                stream.seek(max(0, p.stat().st_size-3072))
                logs[str(p.relative_to(root))] = stream.read().decode(errors='replace').splitlines()[-8:]
    result_path = root/'full/evaluation/metrics.json'
    result = read(result_path) if result_path.exists() else None
    if status == 'complete':
        if not result or not (root/'full/comparison/index.html').is_file():
            warnings.append('Missing full metrics or gallery despite completion marker')
        elif (result['images'], result['queries']) != (2661, 5232) or len(result['methods']) != 5:
            warnings.append('Full evaluation counts do not match the five-method protocol')
        doc = Path(config['project'])/'docs/research/实验结果.md'
        if not doc.is_file() or '<!-- point-union-suite-start -->' not in doc.read_text():
            warnings.append('Result document block is missing')
    return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'status': status, 'terminal': terminal,
            'state': state, 'supervisor_alive': alive, 'active_workers': workers, 'progress': progress,
            'log_tails': logs, 'result': result, 'warnings': warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--no-notify', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    directory = Path(config['experiment'])/'monitor'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'check.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (directory/'finished.json').exists():
            return
        snapshot = inspect(config)
        write(directory/'latest.json', snapshot)
        with (directory/'checks.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot, ensure_ascii=False)+'\n')
        if not args.no_notify:
            prompt = (
                '这是用户授权的每小时 Valen 五方法全量重跑检查。'
                f'读取 {directory / "latest.json"}，核对进程、日志和各阶段进度，用中文简要汇报。'
                '本轮前景方案为内部中心点／互斥区域并集，只有 P(yes)>96% 接收；后续功能规则不变。'
                '五组为原始填色、仅描边、质心点、定义调整、沿用76张训练权重的条件SFT。'
                '不要频繁轮询、重复启动、重训或改参数。'
                '若完成，核对 full/evaluation/metrics.json、full/comparison/index.html 和 '
                'AffordanceJEV/docs/research/实验结果.md；每种方法应为2661张图、5232个功能对，检查五组指标与可视化链接。'
                '结果文档由本地流程自动写入；仅作必要收尾核验。若异常，报告具体原因。'
                '仅在完成或异常退出时核对 monitor/finished.json 和定时器停止状态。')
            delivery = subprocess.run([config['codex'], 'queue', '--thread', config['thread_id'], '--message', prompt],
                cwd=config['workspace'], capture_output=True, text=True, timeout=45)
            with (directory/'notifications.jsonl').open('a') as stream:
                stream.write(json.dumps({'checked_utc': snapshot['checked_utc'], 'returncode': delivery.returncode,
                    'stdout': delivery.stdout, 'stderr': delivery.stderr}, ensure_ascii=False)+'\n')
            if delivery.returncode:
                raise RuntimeError('Notification failed: '+delivery.stderr)
            if snapshot['terminal']:
                stopped = subprocess.run(['systemctl', '--user', 'disable', '--now', config['timer_unit']],
                                         capture_output=True, text=True, timeout=15)
                if stopped.returncode:
                    raise RuntimeError(stopped.stderr)
                write(directory/'finished.json', {'status': snapshot['status'], 'checked_utc': snapshot['checked_utc'],
                      'timer_stopped': True, 'final_report_queued': True})
        print(json.dumps({'status': snapshot['status'], 'terminal': snapshot['terminal'],
                          'active_workers': len(snapshot['active_workers'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
