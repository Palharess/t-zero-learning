"""Create (or update in place) the W&B report with the original charts of every A2C run.

    pip install wandb-workspaces        # needs wandb >= 0.22 (separate env is fine)
    python scripts/wandb_report_a2c.py

Run ids and labels come from runs/a2c/<label>/metadata.json (scripts/run_a2c.py).
Prints the report URL. Charts are the raw logged values (no smoothing).
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import wandb_workspaces.reports.v2 as wr

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
ENTITY = os.environ['WANDB_ENTITY']
PROJECT = os.environ.get('WANDB_PROJECT', 'a2c-assignment')

# same palette as scripts/analyze_a2c.py; seed 2 is a lighter tint
COLORS = {
    'baseline': ('#11776f', '#5fb3a8'),
    'num_envs-1': ('#2764eb', '#8aa9f5'), 'num_envs-64': ('#dc663d', '#f0a88d'),
    'num_steps-1': ('#2764eb', '#8aa9f5'), 'num_steps-32': ('#dc663d', '#f0a88d'),
    'num_steps-128': ('#8a4fcf', '#bf9ce8'),
    'ent_coef-0': ('#2764eb', '#8aa9f5'), 'ent_coef-0.1': ('#dc663d', '#f0a88d'),
    'no_baseline': ('#dc663d', '#f0a88d'),
    'num_envs-64-4M': ('#b8471f', '#e08a66'),
    'lunar-cartpole_cfg': ('#9aa5b1', '#9aa5b1'), 'lunar-cartpole_cfg-3M': ('#11776f', '#5fb3a8'),
    'lunar-e16n16-3M': ('#5a7bd8', '#5a7bd8'), 'lunar-g995-3M': ('#dc663d', '#dc663d'),
    'lunar-g995-ent001-3M': ('#8a4fcf', '#8a4fcf'), 'lunar-g999-ent001-3M': ('#b07d12', '#b07d12'),
    'lunar-g999-lr25-e16-5M': ('#2764eb', '#8aa9f5'), 'lunar-g999-e64-5M': ('#dc663d', '#f0a88d'),
    'lunar-g995-lr25-e16-5M': ('#1c1c1c', '#8c8c8c'),
}
SECTIONS = [
    ('Q1 · Número de atores em paralelo (num_envs = 1, 8, 64)',
     'Baseline = 8 ambientes. Os seis runs de 500 mil passos rodaram um de cada vez, para que o SPS seja '
     'comparável. num_envs-64-4M = controle com 4M passos (as 12.500 atualizações do baseline), rodado em '
     'paralelo com outros treinos (SPS não comparável).',
     ['num_envs-1', 'baseline', 'num_envs-64', 'num_envs-64-4M'],
     [('charts/episodic_return_mean_last100', False), ('losses/policy_loss', False), ('charts/SPS', True)]),
    ('Q2 · Horizonte do retorno de n passos (num_steps = 1, 5, 32, 128)',
     'Baseline = 5 (t_max do A3C). O lote muda com n: 8, 40, 256 e 1.024 transições.',
     ['num_steps-1', 'baseline', 'num_steps-32', 'num_steps-128'],
     [('charts/episodic_return_mean_last100', False), ('losses/value_loss', True),
      ('losses/explained_variance', False)]),
    ('Q3 · Coeficiente de entropia (ent_coef = 0, 0,01, 0,1)',
     'Baseline = 0,01.',
     ['ent_coef-0', 'baseline', 'ent_coef-0.1'],
     [('charts/episodic_return_mean_last100', False), ('losses/entropy', False)]),
    ('Q4 · Ablação do baseline (use_baseline = true / false)',
     'Sem baseline, o peso do gradiente de política é o retorno R em vez da vantagem R − V(s).',
     ['baseline', 'no_baseline'],
     [('charts/episodic_return_mean_last100', False), ('charts/advantage_std', True),
      ('charts/advantage_mean', False), ('losses/entropy', False)]),
    ('Extra · LunarLander-v3',
     'Exploração com uma seed (3M passos) a partir da config do CartPole; depois duas correções com duas seeds '
     'e 5M passos (γ = 0,999, ent_coef = 0,001): LR 2,5e-4 com 16 ambientes e 64 ambientes com LR 7e-4; '
     'por fim, γ = 0,995 e ent_coef = 0,001 com LR 2,5e-4 (16 ambientes, 5M passos, duas seeds).',
     ['lunar-cartpole_cfg', 'lunar-cartpole_cfg-3M', 'lunar-e16n16-3M', 'lunar-g995-3M', 'lunar-g995-ent001-3M',
      'lunar-g999-ent001-3M', 'lunar-g999-lr25-e16-5M', 'lunar-g999-e64-5M', 'lunar-g995-lr25-e16-5M'],
     [('charts/episodic_return_mean_last100', False), ('charts/episodic_length_mean_last100', False),
      ('losses/entropy', False), ('losses/explained_variance', False)]),
]


def runs_by_label():
    out = {}
    for meta_path in sorted((ROOT / 'runs' / 'a2c').glob('*/metadata.json')):
        meta = json.loads(meta_path.read_text())
        out[meta['label']] = meta
    return out


def section_blocks(title, text, configs, metrics, runs):
    labels = [f'{c}-s{s}' for c in configs for s in (1, 2) if f'{c}-s{s}' in runs]
    settings = {runs[l]['wandb_id']: wr.RunSettings(color=COLORS[l.rsplit('-s', 1)[0]][int(l[-1]) - 1])
                for l in labels}
    runset = wr.Runset(entity=ENTITY, project=PROJECT, name=title.split(' ·')[0],
                       filters=f"Config('label') in {labels!r}", run_settings=settings)
    panels = [wr.LinePlot(title=m, x='Step', y=[m], log_y=log_y, smoothing_factor=0.0)
              for m, log_y in metrics]
    ids = ', '.join(f'{l}: {runs[l]["wandb_id"]}' for l in labels)
    return [wr.H2(title), wr.P(text), wr.P(f'Runs (nome: ID): {ids}.'),
            wr.PanelGrid(runsets=[runset], panels=panels)]


def main(extra_sections=(), dry_run=False):
    runs = runs_by_label()
    blocks = [wr.P('Atividade A2C (Gradientes de Política). Gabriel Palhares Siqueira · '
                   'palhares@discente.ufg.br. Q1–Q4: CartPole-v1, 500.000 passos, seeds 1 e 2, uma '
                   'alteração por vez sobre configs/a2c_cartpole.yml. Extra: LunarLander-v3. '
                   'Cor = configuração; seed 2 em tom mais claro. Valores brutos, sem suavização.')]
    for section in [*SECTIONS, *extra_sections]:
        blocks += section_blocks(*section, runs)
    saved = ROOT / 'reports' / 'a2c' / 'wandb-report.json'
    if saved.exists():  # update the published report in place, keeping its URL
        report = wr.Report.from_url(json.loads(saved.read_text())['url'])
        report.blocks = blocks
    else:
        report = wr.Report(entity=ENTITY, project=PROJECT,
                           title='A2C - CartPole e LunarLander - Gabriel Palhares Siqueira',
                           description='Gráficos originais dos runs do relatório de A2C.',
                           blocks=blocks, width='fluid')
    if dry_run:
        print(f'{len(blocks)} blocks; not saved')
        return
    report.save()
    url = report.url.replace(chr(92), '/')  # wandb-workspaces joins the URL with os.path on Windows
    print(url)
    saved.write_text(json.dumps({'url': url}, indent=2))


if __name__ == '__main__':
    import sys
    main(dry_run='--dry-run' in sys.argv)
