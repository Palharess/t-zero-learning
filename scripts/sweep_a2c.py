"""Run the predeclared A2C configurations, two seeds each.

Examples (from the repository root):
    .venv/Scripts/python scripts/sweep_a2c.py --jobs 1 --only baseline num_envs-1 num_envs-64
    .venv/Scripts/python scripts/sweep_a2c.py --jobs 4

Each run goes through scripts/run_a2c.py (label = <config>-s<seed>). Completed
runs are skipped; an incomplete run folder is an error rather than overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# name: (W&B tags = questions, config file, overrides)
CONFIGS = {
    'baseline': ('Q1,Q2,Q3,Q4', 'a2c_cartpole', []),
    'num_envs-1': ('Q1', 'a2c_cartpole', ['num_envs=1']),
    'num_envs-64': ('Q1', 'a2c_cartpole', ['num_envs=64']),
    'num_steps-1': ('Q2', 'a2c_cartpole', ['a2c.num_steps=1']),
    'num_steps-32': ('Q2', 'a2c_cartpole', ['a2c.num_steps=32']),
    'num_steps-128': ('Q2', 'a2c_cartpole', ['a2c.num_steps=128']),
    'ent_coef-0': ('Q3', 'a2c_cartpole', ['a2c.ent_coef=0.0']),
    'ent_coef-0.1': ('Q3', 'a2c_cartpole', ['a2c.ent_coef=0.1']),
    'no_baseline': ('Q4', 'a2c_cartpole', ['a2c.use_baseline=false']),
}


def run(job):
    name, seed = job
    tags, config, overrides = CONFIGS[name]
    label = f'{name}-s{seed}'
    folder = ROOT / 'runs' / 'a2c' / label
    metrics = folder / 'metrics.jsonl'
    if metrics.exists() and 'eval/mean_return' in metrics.read_text():
        return label, 'already complete'
    if folder.exists():
        raise RuntimeError(f'{label}: incomplete run folder {folder} — inspect it before rerunning')
    command = [sys.executable, 'scripts/run_a2c.py', '--label', label, '--seed', str(seed),
               '--config', config, '--group', name, '--tags', tags, '--override', *overrides]
    logdir = ROOT / 'logs' / 'a2c'
    logdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'}
    with (logdir / f'{label}.log').open('w', encoding='utf-8') as out:
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'{label} failed: see {logdir / (label + ".log")}')
    return label, 'complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=4, help='runs executed at the same time')
    parser.add_argument('--only', nargs='*', default=list(CONFIGS), help='configuration names')
    parser.add_argument('--seeds', nargs='*', type=int, default=[1, 2])
    opts = parser.parse_args()
    unknown = set(opts.only) - set(CONFIGS)
    if unknown:
        sys.exit(f'unknown configurations: {sorted(unknown)}; available: {list(CONFIGS)}')
    jobs = [(name, seed) for name in opts.only for seed in opts.seeds]
    failed = []
    with ThreadPoolExecutor(max_workers=opts.jobs) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                label, state = future.result()
                print(f'{label}: {state}', flush=True)
            except RuntimeError as error:
                failed.append(futures[future])
                print(error, flush=True)
    if failed:
        sys.exit(f'{len(failed)} run(s) failed: {failed}')
