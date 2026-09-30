"""Compare trained critics with the Monte Carlo returns of their own final policy.

    .venv/Scripts/python scripts/critic_check_a2c.py num_steps-1-s1 baseline-s1 ...

For each run label, loads runs/a2c/<label>/<run>/model.pt, rolls out the final
(stochastic) policy in fresh environments and compares V(s) with the realised
discounted return G_t, using the harness's convention (an episode end —
termination or truncation — stops the return). Also reports the explained
variance against the 1-step target r + gamma * V(s'), which is what the
training logs show for num_steps=1. Writes reports/a2c/critic-check.json.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def explained_variance(pred, target):
    return float(1 - (target - pred).var() / target.var())


def check(label, steps_per_env=4000, num_envs=8, seed=10_000):
    import gymnasium as gym
    import numpy as np
    import torch
    from core.config_loader import load_config
    from envs import make_env
    from envs.wrappers import discrete_control_wrappers
    from networks.discrete_actor_critic import DiscreteActorCritic

    run_dir = next(p.parent for p in (ROOT / 'runs' / 'a2c' / label).glob('*/model.pt'))
    args, _ = load_config(str(run_dir / 'config.yml'))
    gamma = args.algo.gamma
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, False, 'check', gamma,
                                              wrappers=discrete_control_wrappers)
                                     for i in range(num_envs)])
    agent = DiscreteActorCritic(envs, args.agent.activation, args.agent.hidden_layers_size)
    agent.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=True))
    agent.eval()
    torch.manual_seed(seed)

    T, N = steps_per_env, num_envs
    values, rewards, dones, valid = (np.zeros((T + 1, N)) for _ in range(4))
    obs, _ = envs.reset(seed=seed)
    autoreset = np.zeros(N)
    with torch.no_grad():
        for t in range(T + 1):
            x = torch.as_tensor(obs, dtype=torch.float32)
            action, _, _, value = agent.get_action_and_value(x)
            values[t], valid[t] = value.squeeze(1).numpy(), 1 - autoreset
            if t == T:
                break
            obs, r, term, trunc, _ = envs.step(action.numpy())
            rewards[t], dones[t] = r, np.logical_or(term, trunc)
            autoreset = dones[t]
    envs.close()

    # Monte Carlo returns; transitions after each env's last episode end are incomplete
    G = np.zeros((T, N))
    complete = np.zeros((T, N), dtype=bool)
    running, seen_end = np.zeros(N), np.zeros(N, dtype=bool)
    for t in reversed(range(T)):
        seen_end |= dones[t].astype(bool)
        running = rewards[t] + gamma * (1 - dones[t]) * running
        G[t], complete[t] = running, seen_end
    one_step = rewards[:T] + gamma * (1 - dones[:T]) * values[1:]
    keep = complete & (valid[:T] == 1)
    v, g, r1 = values[:T][keep], G[keep], one_step[keep]
    episodes = int(dones[:T][valid[:T] == 1].sum())
    return {
        'label': label, 'transitions': int(keep.sum()), 'episodes': episodes,
        'mean_episode_length': float(keep.sum() / max(episodes, 1)),
        'mean_V': float(v.mean()), 'mean_G': float(g.mean()), 'bias_V_minus_G': float((v - g).mean()),
        'rmse_V_vs_G': float(np.sqrt(((v - g) ** 2).mean())),
        'ev_vs_monte_carlo': explained_variance(v, g),
        'ev_vs_one_step_target': explained_variance(v, r1),
    }


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    results = [check(label) for label in sys.argv[1:]]
    out = ROOT / 'reports' / 'a2c' / 'critic-check.json'
    out.write_text(json.dumps(results, indent=2))
    for r in results:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
