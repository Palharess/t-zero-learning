"""Run the unchanged training harness with a local copy of W&B metrics.

Example (from repository root):
    .venv/bin/python scripts/run_assignment.py --label baseline-s1 --seed 1
W&B mode is controlled by WANDB_MODE; offline runs can be synced later.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('WANDB_PROJECT', 'dqn-assignment')

import torch
import wandb
from core.config_loader import load_config, apply_overrides
from algorithms.dqn import DQN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    opts = parser.parse_args()
    torch.set_num_threads(1)
    folder = ROOT / 'runs' / 'assignment' / opts.label
    folder.mkdir(parents=True, exist_ok=False)
    args, _ = load_config(str(ROOT / 'configs/dqn_cartpole.yml'))
    args.exp_name = 'assignment/' + opts.label
    apply_overrides(args, [f'seed={opts.seed}', 'capture_video=false', *opts.override])
    agent = DQN(args)
    agent.initialize()
    original_log = wandb.log
    with (folder / 'metrics.jsonl').open('w') as handle:
        def log(data, *pos, **kw):
            handle.write(json.dumps({'step': kw.get('step'), **data}) + '\n')
            handle.flush()
            return original_log(data, *pos, **kw)
        wandb.log = log
        wandb.run.name = opts.label
        metadata = {'label': opts.label, 'seed': opts.seed,
                    'overrides': opts.override, 'wandb_id': wandb.run.id,
                    'wandb_mode': wandb.run.settings.mode,
                    'wandb_dir': wandb.run.dir, 'run_dir': str(agent.run_dir)}
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        try:
            agent.train()
        finally:
            wandb.finish()
            wandb.log = original_log

if __name__ == '__main__':
    main()
