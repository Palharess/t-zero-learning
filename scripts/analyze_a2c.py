"""Per-run summaries and report figures, from the local copies of the W&B metrics.

    python scripts/analyze_a2c.py        # needs numpy + matplotlib

Reads runs/a2c/<label>/{metadata.json,metrics.jsonl} (written by
scripts/run_a2c.py; the same rows that were logged to W&B) and writes
reports/a2c/{results.csv,results-summary.json,q1.png,...}.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs' / 'a2c'
OUT = ROOT / 'reports' / 'a2c'
BLOCK = 5000          # steps per averaging block in the curves (raw curves are on W&B)
INK = '#14293d'

RET = 'charts/episodic_return_mean_last100'
METRICS = [RET, 'losses/policy_loss', 'losses/value_loss', 'losses/explained_variance',
           'losses/entropy', 'losses/grad_norm', 'charts/advantage_mean',
           'charts/advantage_std', 'charts/SPS', 'charts/episodic_length_mean_last100']

# configuration name -> (legend label, color); the baseline is teal everywhere
STYLE = {
    'baseline': ('baseline', '#11776f'),
    'num_envs-1': ('1 ambiente', '#2764eb'),
    'num_envs-64': ('64 ambientes', '#dc663d'),
    'num_envs-64-4M': ('64 amb., 4M passos (controle)', '#dc663d'),
    'num_steps-1': ('n = 1', '#2764eb'),
    'num_steps-32': ('n = 32', '#dc663d'),
    'num_steps-128': ('n = 128', '#8a4fcf'),
    'ent_coef-0': ('ent_coef = 0', '#2764eb'),
    'ent_coef-0.1': ('0,1', '#dc663d'),
    'no_baseline': ('sem baseline', '#dc663d'),
    # Extra — LunarLander-v3 (config names from scripts/sweep_a2c.py)
    'lunar-cartpole_cfg': ('config do CartPole (500 mil)', '#9aa5b1'),
    'lunar-cartpole_cfg-3M': ('config do CartPole (3M)', '#11776f'),
    'lunar-e16n16-3M': ('16 amb. × n=16, 3M', '#2764eb'),
    'lunar-e32n8-3M': ('32 amb. × n=8, 3M', '#dc663d'),
    'lunar-g995-3M': ('γ = 0,995', '#dc663d'),
    'lunar-g995-ent001-3M': ('γ 0,995, ent 0,001 (3M)', '#8a4fcf'),
    'lunar-g999-ent001-3M': ('γ 0,999, ent 0,001 (3M)', '#b07d12'),
    'lunar-g999-lr25-e16-5M': ('γ 0,999 + LR 2,5e-4', '#2764eb'),
    'lunar-g999-e64-5M': ('γ 0,999 + 64 amb.', '#dc663d'),
    'lunar-g995-lr25-e16-5M': ('γ 0,995 + LR 2,5e-4', '#1c1c1c'),
}
LUNAR = ['lunar-cartpole_cfg-3M', 'lunar-g995-ent001-3M', 'lunar-g999-lr25-e16-5M', 'lunar-g999-e64-5M',
         'lunar-g995-lr25-e16-5M']
BASELINE_LABEL = {'Q1': '8 ambientes (baseline)', 'Q2': 'n = 5 (baseline)',
                  'Q3': '0,01 (baseline)', 'Q4': 'com baseline'}
QUESTIONS = {
    'Q1': ['num_envs-1', 'baseline', 'num_envs-64'],
    'Q2': ['num_steps-1', 'baseline', 'num_steps-32', 'num_steps-128'],
    'Q3': ['ent_coef-0', 'baseline', 'ent_coef-0.1'],
    'Q4': ['baseline', 'no_baseline'],
}
SEEDS = (1, 2)
DASH = {1: '-', 2: '--'}
CONTROL_DASH = {1: ':', 2: '-.'}
CONTROLS = {'num_envs-64-4M'}


def dash(name, seed):
    return (CONTROL_DASH if name in CONTROLS else DASH)[seed]


def load_run(folder):
    meta = json.loads((folder / 'metadata.json').read_text())
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    train = [r for r in rows if 'losses/policy_loss' in r]
    evals = [r for r in rows if 'eval/mean_return' in r]
    series = {}
    for key in METRICS + ['time']:
        points = [(r['step'], r[key]) for r in train if r.get(key) is not None]
        steps, values = (np.array(v, dtype=float) for v in zip(*points)) if points else (np.array([]), np.array([]))
        series[key] = (steps, values)
    return {'meta': meta, 'series': series, 'eval': evals[-1] if evals else None}


def load_all():
    runs = {}
    for folder in sorted(RUNS.iterdir()):
        if (folder / 'metrics.jsonl').exists() and (folder / 'metadata.json').exists():
            runs[folder.name] = load_run(folder)
    return runs


def block_mean(steps, values, width=BLOCK, agg=np.nanmean):
    """Aggregate (mean by default) the logged points inside consecutive `width`-step blocks."""
    if steps.size == 0:
        return steps, values
    blocks = np.ceil(steps / width).astype(int)
    keys = np.unique(blocks)
    x = np.array([steps[blocks == k].mean() for k in keys])
    y = np.array([agg(values[blocks == k]) for k in keys])
    return x, y


def summarize(label, run):
    meta, s = run['meta'], run['series']
    total = s[RET][0].max() if s[RET][0].size else 0
    window = lambda key: s[key][1][s[key][0] >= 0.9 * total]   # last 10% of training
    out = {k: meta.get(k) for k in ('label', 'group', 'seed', 'wandb_id', 'num_envs', 'num_steps',
                                    'batch_size', 'num_iterations', 'duration_s')}
    out['config'] = meta.get('group') or label.rsplit('-s', 1)[0]
    if run['eval']:
        out['eval_mean'] = run['eval']['eval/mean_return']
        out['eval_std'] = run['eval']['eval/std_return']
    for key in METRICS:
        vals = window(key)
        out['final_' + key.split('/')[-1]] = float(np.nanmean(vals)) if vals.size else None
    out['max_return_last100'] = float(s[RET][1].max()) if s[RET][1].size else None
    steps, rets = s[RET]
    hit = np.nonzero(rets >= 450)[0]
    out['steps_to_450'] = float(steps[hit[0]]) if hit.size else None
    out['updates_to_450'] = out['steps_to_450'] / meta['batch_size'] if hit.size else None
    at = np.argmin(np.abs(steps - 1500 * meta['batch_size'])) if steps.size else None
    out['return_at_1500_updates'] = float(rets[at]) if at is not None else None
    hit200 = np.nonzero(rets >= 200)[0]
    out['steps_to_200'] = float(steps[hit200[0]]) if hit200.size else None
    t_steps, t_vals = s['time']
    t0 = meta['start_time']
    out['time_to_450_s'] = float(t_vals[t_steps == steps[hit[0]]][0] - t0) if hit.size else None
    pl_steps, pl = s['losses/policy_loss']
    out['policy_loss_std'] = float(np.std(pl))
    out['policy_loss_std_2nd_half'] = float(np.std(pl[pl_steps >= total / 2]))
    out['policy_loss_mean'] = float(np.mean(pl))
    out['policy_loss_frac_small'] = float(np.mean(np.abs(pl) < 0.05))
    out['policy_loss_p99_abs'] = float(np.percentile(np.abs(pl), 99))
    for key, name in (('losses/explained_variance', 'ev'), ('losses/value_loss', 'value_loss'),
                      ('charts/advantage_std', 'advantage_std'), ('charts/advantage_mean', 'advantage_mean'),
                      ('losses/grad_norm', 'grad_norm'), ('losses/entropy', 'entropy')):
        vals = s[key][1]
        out[f'{name}_median'] = float(np.nanmedian(vals)) if vals.size else None
        out[f'{name}_mean'] = float(np.nanmean(vals)) if vals.size else None
    out['frac_clipped'] = float(np.mean(s['losses/grad_norm'][1] > meta.get('max_grad_norm', 0.5)))
    ent_steps, ent = s['losses/entropy']
    low = np.nonzero(ent < 0.1)[0]
    out['steps_entropy_below_0.1'] = float(ent_steps[low[0]]) if low.size else None
    out['sps_final'] = float(s['charts/SPS'][1][-1]) if s['charts/SPS'][1].size else None
    return out


def style_axes(ax, title, xlabel='Passos (mil)'):
    ax.set_title(title, fontsize=9, color=INK, pad=4)
    ax.set_xlabel(xlabel, fontsize=7.8, color='#35506b', labelpad=2)
    ax.tick_params(labelsize=7, colors=INK, length=2.5, pad=2)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def thousands(value, _pos=None):
    """pt-BR integer tick labels: 20000 -> '20.000'."""
    return f'{value:,.0f}'.replace(',', '.')


def log_ticks(ax, ticks):
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.yaxis.set_minor_formatter(NullFormatter())


def symlog_ticks(ax, ticks):
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{v:g}'.replace('-', '−')))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_metric(ax, runs, configs, key, *, smooth=True, xmode='steps', alpha=1.0, lw=1.3,
                width=BLOCK, xscale=1000, agg=np.nanmean):
    for name in configs:
        color = STYLE[name][1]
        for seed in SEEDS:
            run = runs.get(f'{name}-s{seed}')
            if run is None:
                continue
            steps, values = run['series'][key]
            if smooth:
                steps, values = block_mean(steps, values, width, agg)
            if xmode == 'steps':
                x = steps / xscale
            else:  # number of gradient updates so far
                x = steps / run['meta']['batch_size']
            ax.plot(x, values, dash(name, seed), color=color, lw=lw, alpha=alpha)


def legend(fig, question, configs, ncol=None, y=-0.02):
    """One row: a color per configuration, then the seed line styles."""
    handles, labels = [], []
    for name in configs:
        handles.append(Line2D([], [], color=STYLE[name][1], ls=':' if name in CONTROLS else '-', lw=1.6))
        labels.append(BASELINE_LABEL.get(question, 'baseline') if name == 'baseline' else STYLE[name][0])
    for seed in SEEDS:
        handles.append(Line2D([], [], color='#555555', ls=DASH[seed], lw=1.2))
        labels.append(f'seed {seed}')
    fig.legend(handles, labels, loc='lower center', ncol=ncol or len(handles), frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, y), handlelength=2.2, columnspacing=1.3,
               handletextpad=0.5)


def x_steps(ax):
    ax.set_xlim(0, 500)
    ax.set_xticks([0, 250, 500])


def figure_q1(runs):
    cfg = QUESTIONS['Q1']
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 1.75))
    plot_metric(axes[0], runs, cfg, RET)
    style_axes(axes[0], 'Retorno médio (100 ep.)'); x_steps(axes[0]); axes[0].set_ylim(0, 520)
    plot_metric(axes[1], runs, cfg + ['num_envs-64-4M'], RET, xmode='updates')
    style_axes(axes[1], 'Retorno × atualizações', 'Atualizações (log)'); axes[1].set_xscale('log')
    axes[1].set_ylim(0, 520)
    for name in cfg:  # raw values: the noise is the message; one seed keeps it readable
        run = runs.get(f'{name}-s1')
        if run:
            steps, values = run['series']['losses/policy_loss']
            axes[2].plot(steps / 1000, values, '-', color=STYLE[name][1], lw=0.6, alpha=0.85)
    style_axes(axes[2], 'policy_loss (bruto, seed 1)'); x_steps(axes[2])
    axes[2].set_yscale('symlog', linthresh=1); symlog_ticks(axes[2], [-100, -10, -1, 0, 1, 10, 100])
    plot_metric(axes[3], runs, cfg, 'charts/SPS', smooth=False)
    style_axes(axes[3], 'SPS (passos/s)'); x_steps(axes[3]); axes[3].set_yscale('log')
    log_ticks(axes[3], [1000, 2000, 5000, 10000, 20000])
    legend(fig, 'Q1', cfg + (['num_envs-64-4M'] if 'num_envs-64-4M-s1' in runs else []), y=-0.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    return fig


def figure_q2(runs):
    cfg = QUESTIONS['Q2']
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 1.75))
    plot_metric(axes[0], runs, cfg, RET)
    style_axes(axes[0], 'Retorno médio (100 ep.)'); x_steps(axes[0]); axes[0].set_ylim(0, 520)
    plot_metric(axes[1], runs, cfg, 'losses/value_loss', width=25000, agg=np.nanmedian)
    style_axes(axes[1], 'value_loss (mediana)'); x_steps(axes[1]); axes[1].set_yscale('log')
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    plot_metric(axes[2], runs, cfg, 'losses/explained_variance', width=25000, agg=np.nanmedian)
    style_axes(axes[2], 'explained_variance (mediana)'); x_steps(axes[2]); axes[2].set_ylim(-0.3, 1.02)
    legend(fig, 'Q2', cfg, y=-0.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.5)
    return fig


def figure_q3(runs):
    cfg = QUESTIONS['Q3']
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0))
    plot_metric(axes[0], runs, cfg, RET)
    style_axes(axes[0], 'Retorno médio (100 ep.)'); x_steps(axes[0]); axes[0].set_ylim(0, 520)
    plot_metric(axes[1], runs, cfg, 'losses/entropy')
    style_axes(axes[1], 'Entropia da política'); x_steps(axes[1])
    axes[1].axhline(np.log(2), color='#777777', lw=0.8, ls=':')
    axes[1].set_ylim(0, 0.72)
    legend(fig, 'Q3', cfg, y=-0.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0)
    return fig


def figure_q4(runs):
    cfg = QUESTIONS['Q4']
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 1.75))
    plot_metric(axes[0], runs, cfg, RET)
    style_axes(axes[0], 'Retorno médio (100 ep.)'); x_steps(axes[0]); axes[0].set_ylim(0, 520)
    plot_metric(axes[1], runs, cfg, 'charts/advantage_std')
    style_axes(axes[1], 'advantage_std'); x_steps(axes[1]); axes[1].set_yscale('log')
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    plot_metric(axes[2], runs, cfg, 'charts/advantage_mean')
    style_axes(axes[2], 'advantage_mean'); x_steps(axes[2]); axes[2].set_yscale('symlog', linthresh=1)
    symlog_ticks(axes[2], [-10, -1, 0, 1, 10, 100])
    plot_metric(axes[3], runs, cfg, 'losses/entropy')
    style_axes(axes[3], 'Entropia da política'); x_steps(axes[3]); axes[3].set_ylim(0, 0.72)
    legend(fig, 'Q4', cfg, y=-0.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    return fig


def figure_lunar(runs, configs=None):
    cfg = [c for c in (configs or LUNAR) if f'{c}-s1' in runs]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 1.75))
    kw = dict(width=25000, xscale=1e6)
    plot_metric(axes[0], runs, cfg, RET, **kw)
    axes[0].axhline(200, color='#777777', lw=0.8, ls=':')
    style_axes(axes[0], 'Retorno médio (100 ep.)', 'Passos (milhões)')
    plot_metric(axes[1], runs, cfg, 'charts/episodic_length_mean_last100', **kw)
    axes[1].axhline(1000, color='#777777', lw=0.8, ls=':')
    style_axes(axes[1], 'Duração do episódio (100 ep.)', 'Passos (milhões)'); axes[1].set_ylim(0, 1050)
    plot_metric(axes[2], runs, cfg, 'losses/entropy', **kw)
    axes[2].axhline(np.log(4), color='#777777', lw=0.8, ls=':')
    style_axes(axes[2], 'Entropia da política', 'Passos (milhões)')
    plot_metric(axes[3], runs, cfg, 'losses/explained_variance', agg=np.nanmedian, **kw)
    style_axes(axes[3], 'explained_variance (mediana)', 'Passos (milhões)'); axes[3].set_ylim(-0.2, 1.02)
    legend(fig, 'Extra', cfg, y=-0.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_all()
    summary = [summarize(label, run) for label, run in runs.items()]
    (OUT / 'results-summary.json').write_text(json.dumps(summary, indent=2))
    with (OUT / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for name, make in (('q1', figure_q1), ('q2', figure_q2), ('q3', figure_q3), ('q4', figure_q4),
                       ('lunar', figure_lunar)):
        fig = make(runs)
        fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f'{len(runs)} runs summarized; figures in {OUT}')


if __name__ == '__main__':
    main()
