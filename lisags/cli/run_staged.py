"""Run the final 30k+8k recipe for one scene, then render/evaluate official test.

Family overrides and the appearance handoff match PORT-GS's final manifest.
Single scenes use CUDA_VISIBLE_DEVICES; --gpus runs a scene queue with one job per GPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import zip_longest
import os
from queue import Queue, Empty
from threading import Lock
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def train_command(options):
    """Serialize the canonical JSON fields using the training CLI."""
    command = [sys.executable, '-m', 'lisags.training.trainer']
    for key, value in options.items():
        flag = '--' + key
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif key == 'absgrad':
                command.append('--no-absgrad')
        elif isinstance(value, list):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    return command


def stage_commands(recipe, family, scene, output):
    """Return the original geometry, frozen appearance, and fixed-camera test commands."""
    settings = recipe['families'][family]
    options = {**recipe['train'], **settings['train'],
               'scene': str(Path(recipe['data_root']) / family / scene),
               'output': str(output / 'geometry')}
    geometry = train_command(options)
    appearance = train_command({**options, **recipe['appearance_train'],
                                'output': str(output / 'appearance'),
                                'init-checkpoint': str(output / 'geometry/last.pt')})
    evaluation = [sys.executable, '-m', 'lisags.evaluation.evaluator', str(output / 'appearance/last.pt'),
                  '--split', 'test', '--lpips', '--save-all', '--output', str(output / 'appearance/test')]
    return [('geometry', geometry), ('appearance', appearance), ('test', evaluation)]


def save_source(output):
    """Save project code and dependency pins without copying earlier experiment outputs."""
    with tarfile.open(output / 'source.tar', 'w') as archive:
        for pattern in ('lisags/**/*.py', 'configs/*.json', 'pyproject.toml', '.gitmodules',
                        'third_party/README.md', 'scripts/*.sh'):
            for path in sorted(ROOT.glob(pattern)):
                archive.add(path, arcname=str(path.relative_to(ROOT)))


def run_batch(recipe, config_path, families, gpus, output, dry_run):
    """Drain an interleaved family queue, keeping each physical GPU on one scene."""
    groups = [[(family, scene) for scene in recipe['families'][family]['scenes']]
              for family in families]
    jobs = []
    for group in zip_longest(*groups):
        for item in group:
            if item is None:
                continue
            family, scene = item
            directory = output / family / scene
            command = [sys.executable, '-m', 'lisags.cli.run_staged', '--config', str(config_path),
                       '--data-root', recipe['data_root'], '--family', family, '--scene', scene,
                       '--output', str(directory)]
            jobs.append(dict(family=family, scene=scene, output=str(directory), command=command,
                             state='pending', gpu=None, pid=None))
    if dry_run:
        print(json.dumps(dict(gpus=gpus, jobs=jobs, recipe=recipe), indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    save_source(output)
    # Children use the saved recipe, so editing the canonical config cannot alter queued jobs.
    for job in jobs:
        job['command'][4] = str(output / 'recipe.json')
    state = dict(gpus=gpus, jobs=jobs)
    lock = Lock()
    queue = Queue()
    for job in jobs:
        queue.put(job)

    def save_status():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        (output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

    def progress():
        # Evidence contains actual process relationships, GPU occupancy and training progress.
        processes = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,args='], text=True).splitlines()
        with lock:
            roots = {job['pid'] for job in jobs if job['state'] == 'running'}
            selected = []
            for _ in range(3):  # scene runner -> trainer/evaluator -> any helper process
                for line in processes:
                    pid, parent, _ = line.strip().split(None, 2)
                    if int(pid) in roots or int(parent) in roots:
                        roots.add(int(pid))
                        if line not in selected:
                            selected.append(line)
            active = []
            for job in jobs:
                if job['state'] != 'running':
                    continue
                histories = list(Path(job['output']).glob('*/history.jsonl'))
                latest = max(histories, key=lambda path: path.stat().st_mtime) if histories else None
                active.append(dict(family=job['family'], scene=job['scene'], gpu=job['gpu'],
                                   history=str(latest) if latest else None,
                                   last_record=latest.read_text().splitlines()[-1] if latest and latest.stat().st_size else None))
            snapshot = dict(time=datetime.now(timezone.utc).isoformat(), processes=selected, active=active,
                            gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                                         '--format=csv,noheader'], text=True))
            with (output / 'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(snapshot) + '\n')

    def worker(gpu):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return
            log = output / f"{job['family']}_{job['scene']}.log"
            with log.open('w') as handle:
                process = subprocess.Popen(job['command'], cwd=ROOT, env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT)
                with lock:
                    job.update(state='running', gpu=gpu, pid=process.pid,
                               started_at=datetime.now(timezone.utc).isoformat())
                    save_status()
                code = process.wait()
            with lock:
                job.update(state='completed' if code == 0 else 'failed', returncode=code,
                           finished_at=datetime.now(timezone.utc).isoformat())
                if code == 0:
                    report = json.loads((Path(job['output']) / 'appearance/test/metrics.json').read_text())
                    job['metrics'] = report['metrics']
                    job['evaluated_frames'] = report['evaluated_frames']
                save_status()
            if code:
                raise subprocess.CalledProcessError(code, job['command'])

    save_status()
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        pending = {executor.submit(worker, gpu) for gpu in gpus}
        while pending:
            done, pending = wait(pending, timeout=3600)
            progress()
            for future in done:
                future.result()
    def average(rows):
        return {key: sum(row['metrics'][key] for row in rows) / len(rows)
                for key in ('PSNR', 'SSIM', 'LPIPS')}
    summary = dict(scene_count=len(jobs), frames=sum(job['evaluated_frames'] for job in jobs),
                   overall=average(jobs), families={family: average([job for job in jobs if job['family'] == family])
                                                   for family in families})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/staged.json')
    parser.add_argument('--data-root', type=Path, help='Dataset root containing family/scene folders')
    parser.add_argument('--family', help='Dataset family; omitted in batch mode runs all families')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--scene', help='Run a single scene in --family')
    mode.add_argument('--gpus', nargs='+', type=int, help='Run the configured scene queue, one job per physical GPU')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running them')
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text())
    recipe['data_root'] = str((args.data_root or Path(recipe['data_root'])).resolve())
    output = args.output.resolve()
    if args.gpus is not None:
        families = [args.family] if args.family else list(recipe['families'])
        run_batch(recipe, args.config.resolve(), families, args.gpus, output, args.dry_run)
        return
    if args.family is None:
        parser.error('--scene requires --family')
    commands = stage_commands(recipe, args.family, args.scene, output)
    for _, command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    (output / 'commands.json').write_text(json.dumps(dict(commands), indent=2) + '\n')
    save_source(output)
    for phase, command in commands:
        with (output / f'{phase}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
