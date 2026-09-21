"""Run two seeds for each predeclared configuration (up to 3 processes).

Run baseline-s1 separately first; this script skips it. Existing completed runs
are skipped; existing incomplete runs raise an error rather than overwrite data.
"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'baseline': [],
    'target-1': ['dqn.target_network_frequency=1'],
    'target-10000': ['dqn.target_network_frequency=10000'],
    'buffer-128': ['dqn.buffer_size=128'],
    'buffer-100000': ['dqn.buffer_size=100000'],
    'lr-small': ['dqn.learning_rate=0.000025'],
    'lr-large': ['dqn.learning_rate=0.0025'],
}

def run(item):
    name, seed, overrides = item
    label = f'{name}-s{seed}'
    metrics = ROOT / 'runs' / 'assignment' / label / 'metrics.jsonl'
    if metrics.exists() and 'eval/mean_return' in metrics.read_text():
        return label, 'already complete'
    command = [sys.executable, 'scripts/run_assignment.py', '--label', label,
               '--seed', str(seed), '--override', *overrides]
    logdir = ROOT / 'logs' / 'assignment'
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / f'{label}.log').open('w') as out:
        result = subprocess.run(command, cwd=ROOT, stdout=out,
                                stderr=subprocess.STDOUT, env=os.environ.copy())
    if result.returncode:
        raise RuntimeError(f'{label} failed: see {logdir / (label + ".log")}')
    print(f'{label}: complete', flush=True)
    return label, 'complete'

if __name__ == '__main__':
    jobs = [(name, seed, overrides) for name, overrides in CONFIGS.items()
            for seed in (1, 2) if (name, seed) != ('baseline', 1)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for label, state in pool.map(run, jobs):
            print(label, state, flush=True)
