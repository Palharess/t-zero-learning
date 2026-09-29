"""Run the unchanged A2C harness as one labelled W&B run, keeping a local copy of its metrics.

Example (from the repository root):
    .venv/Scripts/python scripts/run_a2c.py --label baseline-s1 --seed 1
    .venv/Scripts/python scripts/run_a2c.py --label num_envs-1-s1 --seed 1 --override num_envs=1

W&B settings come from .env (WANDB_ENTITY, WANDB_PROJECT, WANDB_API_KEY);
WANDB_MODE=offline also works and can be synced later. Everything a run logs
to W&B is also written to runs/a2c/<label>/metrics.jsonl, and the run id and
settings to runs/a2c/<label>/metadata.json.

Heavy imports live inside main(): on Windows, AsyncVectorEnv workers are
spawned and re-import this file, so they should not import torch/wandb.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, help='W&B run name and local folder name')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', default='a2c_cartpole', help='config name in configs/')
    parser.add_argument('--group', default='', help='W&B group (configuration without seed)')
    parser.add_argument('--tags', default='', help='comma-separated W&B tags')
    parser.add_argument('--override', nargs='*', default=[])
    opts = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    os.environ.setdefault('WANDB_PROJECT', 'a2c-assignment')
    if opts.group:
        os.environ['WANDB_RUN_GROUP'] = opts.group
    if opts.tags:
        os.environ['WANDB_TAGS'] = opts.tags

    import gymnasium
    import torch
    import wandb
    from core.config_loader import load_config, apply_overrides
    from algorithms.a2c import A2C

    torch.set_num_threads(1)
    folder = ROOT / 'runs' / 'a2c' / opts.label
    folder.mkdir(parents=True, exist_ok=False)
    args, _ = load_config(str(ROOT / 'configs' / f'{opts.config}.yml'))
    args.exp_name = 'a2c/' + opts.label
    overrides = [f'seed={opts.seed}', 'capture_video=false', *opts.override]
    apply_overrides(args, overrides)

    agent = A2C(args)
    agent.initialize()
    wandb.run.name = opts.label
    wandb.config.update({'label': opts.label, 'config_name': opts.config,
                         'overrides': overrides}, allow_val_change=True)
    metadata = {
        'label': opts.label, 'seed': opts.seed, 'config': opts.config,
        'group': opts.group, 'tags': opts.tags, 'overrides': overrides,
        'wandb_id': wandb.run.id, 'wandb_url': wandb.run.get_url(),
        'wandb_mode': wandb.run.settings.mode, 'run_dir': str(agent.run_dir),
        'num_envs': args.num_envs, 'num_steps': args.algo.num_steps,
        'batch_size': args.batch_size, 'num_iterations': args.num_iterations,
        'python': platform.python_version(), 'torch': torch.__version__,
        'gymnasium': gymnasium.__version__, 'wandb': wandb.__version__,
        'platform': platform.platform(), 'start_time': time.time(),
    }
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    original_log = wandb.log
    with (folder / 'metrics.jsonl').open('w') as handle:
        def log(data, *pos, **kw):
            row = {'step': kw.get('step'), 'time': time.time(), **data}
            handle.write(json.dumps(row, default=float) + '\n')
            handle.flush()
            return original_log(data, *pos, **kw)

        wandb.log = log
        try:
            agent.train()
        finally:
            wandb.finish()
            wandb.log = original_log
    metadata['end_time'] = time.time()
    metadata['duration_s'] = metadata['end_time'] - metadata['start_time']
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
