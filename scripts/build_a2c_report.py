"""Build reports/a2c/Relatorio_A2C_Gabriel_Palhares_Siqueira.pdf (A4, at most 3 pages).

    python scripts/analyze_a2c.py        # summaries + figures first
    python scripts/build_a2c_report.py   # needs reportlab + matplotlib (fonts) + pillow + pypdf

Every number in the text is read from reports/a2c/results-summary.json (and
critic-check.json), so the report cannot drift from the logged metrics.
"""
import json
import sys
from pathlib import Path

from reportlab.platypus import KeepTogether, Paragraph, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_layout import CONTENT_W, build, figure, link, styles, table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'a2c'
PDF = OUT / 'Relatorio_A2C_Gabriel_Palhares_Siqueira.pdf'
AUTHOR = 'Gabriel Palhares Siqueira'
EMAIL = 'palhares@discente.ufg.br'
FORK = 'https://github.com/Palharess/t-zero-learning'
DATE = '29/09/2026'
_wandb = OUT / 'wandb-report.json'
WANDB = json.loads(_wandb.read_text())['url'] if _wandb.exists() else ''

S = {r['label']: r for r in json.loads((OUT / 'results-summary.json').read_text())}
CRITIC = {r['label']: r for r in json.loads((OUT / 'critic-check.json').read_text())}
ST = styles(body_size=8.3, leading=10.0)


# ----------------------------------------------------------------------------- number helpers

def v(label, key):
    return S[label][key]


def num(x, nd=1):
    """pt-BR number: 1234.5 -> '1.234,5'."""
    text = f'{x:,.{nd}f}'
    return text.replace(',', '§').replace('.', ',').replace('§', '.')


def both(config, key, nd=1, fn=None, sep=' e '):
    vals = [v(f'{config}-s{s}', key) for s in (1, 2)]
    return sep.join((fn or (lambda x: num(x, nd)))(x) for x in vals)


def ev(label):
    if 'eval_mean' not in S.get(label, {}):
        return '—'
    return f"{num(v(label, 'eval_mean'))} ± {num(v(label, 'eval_std'))}"


def kilo(steps, unit=' mil'):
    return '—' if steps is None else f'{num(steps / 1000, 0)}{unit}'


def avg(config, key, nd=1):
    return num(sum(v(f'{config}-s{s}', key) for s in (1, 2)) / 2, nd)


def pair(config, key, fn):
    return ' / '.join(fn(v(f'{config}-s{s}', key)) for s in (1, 2))


def run_ids(configs):
    return '; '.join(f"{c.removeprefix('lunar-')}-s{s}: {v(f'{c}-s{s}', 'wandb_id')}" for c in configs
                     for s in (1, 2) if f'{c}-s{s}' in S)


# ----------------------------------------------------------------------------- flowable helpers

def P(text, style='body'):
    return Paragraph(text, ST[style])


def lead(label, text):
    return P(f'<b>{label}</b> {text}')


def section(title, prediction, sweep, visual, tab, explanation, configs):
    """One question: prediction -> sweep -> figure (+caption) -> table -> explanation -> run ids."""
    out = [KeepTogether([P(title, 'h2'), lead('Previsão anterior aos treinos.', prediction)]),
           lead('Varredura.', sweep), KeepTogether(visual)]
    if tab is not None:
        out += [tab, Spacer(1, 2)]
    out.append(lead('Explicação.', explanation[0]))
    out += [P(t) for t in explanation[1:]]
    out.append(P(f'Runs (nome: ID W&amp;B): {run_ids(configs)}.', 'small'))
    return out


def cols(*fractions, width=CONTENT_W):
    return [width * f for f in fractions]


CAPTION = ('Métricas registradas no W&amp;B, média por blocos de 5.000 passos; cor = configuração, '
           'contínua = seed 1, tracejada = seed 2. Curvas sem agregação no link do W&amp;B.')


# ----------------------------------------------------------------------------- sections

def front():
    links = f"Fork: {link(FORK, 'github.com/Palharess/t-zero-learning')}"
    if WANDB:
        links += f" · Gráficos originais: {link(WANDB, 'relatório no W&amp;B')} (Q1–Q4 e Extra)"
    links += (f" · Previsões, adições feitas durante a execução e reprodução: "
              f"{link(FORK + '/blob/main/EXPERIMENTOS_A2C.md', 'EXPERIMENTOS_A2C.md')}")
    n_cart = sum(1 for l in S if not l.startswith('lunar'))
    n_lunar = sum(1 for l in S if l.startswith('lunar'))
    return [
        P('Actor-Critic Síncrono (A2C) no CartPole', 'title'),
        P(f'{AUTHOR} · {EMAIL} · Atividade individual · {DATE}', 'byline'),
        P(f'<b>Protocolo:</b> {n_cart} treinos no CartPole-v1 (500.000 passos; o controle da Q1 usa 4M) e '
          f'{n_lunar} no LunarLander-v3. Seeds 1 e 2 em toda configuração do CartPole (duas seeds não bastam para '
          f"conclusões universais: o próprio baseline avalia {ev('baseline-s1')} e {ev('baseline-s2')}; a mesma seed "
          'reproduz o treino bit a bit); uma alteração por vez sobre '
          '<i>configs/a2c_cartpole.yml</i> (8 ambientes, n = 5, ent_coef = 0,01, LR 7e-4, γ = 0,99). Avaliação final '
          'do harness: 10 episódios com a política estocástica (média ± DP entre episódios). Os 14 testes de '
          '<i>tests/test_a2c.py</i> passaram. Vídeo desativado; nenhuma outra mudança no harness. '
          '<b>Implementação:</b> R(t) = r(t) + γ(1 − d(t))·R(t+1), de trás para '
          'frente a partir de R(T) = V(s(T)), por coluna; perda −média(log π·A) com '
          'A = R − V(s) (ou R) <i>destacado</i> do grafo — constante para o ator, o crítico só aprende pelo '
          'value_loss; Categorical(logits) com sample()/argmax, log_prob da ação recebida e entropy().', 'small'),
        P(links, 'small'),
    ]


def q1():
    c = ['num_envs-1', 'baseline', 'num_envs-64', 'num_envs-64-4M']
    names = {'num_envs-1': '1 ambiente', 'baseline': '8 (baseline)', 'num_envs-64': '64 ambientes',
             'num_envs-64-4M': '64, 4M (controle)'}
    rows = [['Configuração', 'Atualiz.', 'SPS', 'Tempo (s)', 'Passos até 450',
             'Atualiz. até 450', 'DP policy_loss', 'Avaliação s1', 'Avaliação s2']]
    for name in c:
        a, b = f'{name}-s1', f'{name}-s2'
        if 'eval_mean' not in S.get(b, {}):
            continue
        control = name == 'num_envs-64-4M'
        rows.append([names[name], num(v(a, 'num_iterations'), 0),
                     '—' if control else pair(name, 'sps_final', lambda x: num(x, 0)),
                     '—' if control else pair(name, 'duration_s', lambda x: num(x, 0)),
                     pair(name, 'steps_to_450', lambda x: kilo(x, 'k')),
                     pair(name, 'updates_to_450', lambda x: '—' if x is None else num(x, 0)),
                     pair(name, 'policy_loss_std', lambda x: num(x, 1)), ev(a), ev(b)])
    tab = table(rows, cols(0.155, 0.075, 0.14, 0.085, 0.105, 0.12, 0.09, 0.115, 0.115), ST)
    frac = num(100 * v('num_envs-1-s1', 'policy_loss_frac_small'), 0)
    p99 = num(v('num_envs-1-s1', 'policy_loss_p99_abs'), 0)
    explanation = [
        f'Previsão confirmada. Com 1 ambiente (seed 1), o policy_loss fica ≈ 0 em {frac}% das atualizações e às vezes '
        f"explode (p99 de |policy_loss| = {p99}; DP {both('num_envs-1', 'policy_loss_std')}, contra "
        f"≈{avg('baseline', 'policy_loss_std')} no baseline e ≈{avg('num_envs-64', 'policy_loss_std')} com 64). Com "
        'recompensa sempre +1, 5 passos seguidos de um episódio quase não trazem sinal (vantagens ≈ 0): o gradiente vem '
        'dos raros lotes com fim de episódio, e um único evento decide a direção do passo. O agente segue um '
        'estimador de variância enorme — 100.000 passos, cada um ditado por um ou nenhum evento — e o retorno oscila '
        'sem chegar a 450. É o argumento do A3C: sem replay, um método on-policy descorrelaciona os dados com vários '
        'atores, cujos lotes misturam episódios diferentes da mesma política.',
        'Os dois efeitos se separam no 2º painel. Por atualização, lote maior é melhor: 64 ambientes chegam a 450 com '
        f"{both('num_envs-64-4M', 'updates_to_450', 0)} atualizações (controle), contra "
        f"{both('baseline', 'updates_to_450', 0)} do baseline — mas 8× mais dados por atualização compram só ~2× "
        'menos atualizações. Por passo de ambiente, o baseline vence: em 500 mil passos cabem só 1.562 atualizações com 64 '
        f"ambientes (retorno final {both('num_envs-64', 'final_episodic_return_mean_last100', 0)}), e o baseline chega "
        f"a 450 aos {both('baseline', 'steps_to_450', fn=kilo)} passos.",
        'O paralelismo compra tempo de relógio: SPS de ≈1.000, 6.200 e 18.100; 500 mil passos em '
        f"{num(v('num_envs-1-s1', 'duration_s') / 60, 1)} min, {num(v('baseline-s1', 'duration_s'), 0)} s e "
        f"{num(v('num_envs-64-s1', 'duration_s'), 0)} s. O ganho é sublinear (1→8: 6×; 8→64: 2,9×) porque o laço "
        'principal é serial — uma inferência e uma troca de mensagens com todos os processos por passo — e a máquina '
        f"tem 12 threads. Em tempo até 450, o baseline foi o melhor ({both('baseline', 'time_to_450_s', 0)} s).",
    ]
    caption = (CAPTION + ' 2º painel: x = número de atualizações; pontilhado = controle (s1; s2 em traço-ponto). '
               'policy_loss: valores brutos da seed 1.')
    return section('Q1 · Número de atores em paralelo (num_envs)',
                   'Com 1 ambiente, cada gradiente usa 5 transições seguidas do mesmo episódio: policy_loss muito '
                   'ruidoso e retorno instável, apesar de 8× mais atualizações. Com 64, policy_loss suave, mas 1.562 '
                   'atualizações devem atrasar o aprendizado por passo. SPS crescente, mas sublinear.',
                   'num_envs = 1, 8 (baseline) e 64, duas seeds; lotes de 5, 40 e 320. Os seis runs rodaram um de '
                   'cada vez, sem outro processo, para o SPS ser comparável. Controle adicionado depois: 64 ambientes '
                   'com 4M passos (as 12.500 atualizações do baseline).',
                   [figure(OUT / 'q1.png'), P(caption, 'small')], tab, explanation, c)


def q2():
    c = ['num_steps-1', 'baseline', 'num_steps-32', 'num_steps-128']
    names = {'num_steps-1': 'n = 1', 'baseline': 'n = 5 (baseline)', 'num_steps-32': 'n = 32',
             'num_steps-128': 'n = 128'}
    rows = [['Configuração', 'Atualiz.', 'Passos até 450', 'Atualiz. até 450', 'EV mediana',
             'value_loss médio', 'V̄ − Ḡ (MC)', 'Avaliação s1', 'Avaliação s2']]
    for name in c:
        a, b = f'{name}-s1', f'{name}-s2'
        rows.append([names[name], num(v(a, 'num_iterations'), 0),
                     pair(name, 'steps_to_450', lambda x: kilo(x, 'k')),
                     pair(name, 'updates_to_450', lambda x: '—' if x is None else num(x, 0)),
                     pair(name, 'ev_median', lambda x: num(x, 2)),
                     pair(name, 'value_loss_mean', lambda x: num(x, 0)),
                     ' / '.join(num(CRITIC[x]['bias_V_minus_G'], 1) for x in (a, b)), ev(a), ev(b)])
    tab = table(rows, cols(0.14, 0.075, 0.105, 0.14, 0.1, 0.1, 0.12, 0.11, 0.11), ST)
    c1, c32, c128 = (CRITIC[f'{n}-s1'] for n in ('num_steps-1', 'num_steps-32', 'num_steps-128'))
    explanation = [
        'n = 1 tem o maior viés — o alvo herda todo o erro de V — e a menor variância (só um passo real entra no alvo). '
        'Com n = 128 entram até 128 recompensas reais e o bootstrap pesa γ<super>128</super> ≈ 0,28: pouco viés, '
        'mas o instante da queda ou do truncamento muda a soma inteira. No crítico, o value_loss típico cresce '
        f"ordens de grandeza com n e a EV cai de {both('num_steps-1', 'ev_median', 2)} (n = 1) para ≈ 0 (n = 128). "
        'Previsão confirmada para o crítico.',
        'A EV de n = 1 parece ótima porque alvo e previsão contêm o mesmo V: R = 1 + γV(s′) com V(s′) ≈ V(s), e '
        'qualquer V localmente consistente “explica” o alvo — um V com escala errada (V = αV*) tem erro TD quase '
        'constante (≈ 1 − α) e EV alta para qualquer α. Contra os retornos reais, o crítico de n = 1 prevê '
        f"V̄ = {num(c1['mean_V'])}, o ponto fixo 1/(1 − γ), para Ḡ = {num(c1['mean_G'])} (viés de "
        f"+{num(c1['bias_V_minus_G'])}); o de n = 32 é o mais calibrado ({num(c32['bias_V_minus_G'])}) e o de "
        f"n = 128 subestima ({num(c128['bias_V_minus_G'])}). Com lotes de 8 a EV também é ruidosa, e o value_loss "
        'médio é dominado pelos raros lotes com fim de episódio (o harness trata o truncamento em 500 passos como '
        'término).',
        'O ator sofre com o crítico porque seu peso é A = R − V(s): com n = 128, EV ≈ 0 e advantage_mean ≈ '
        f"+{num(v('num_steps-128-s1', 'advantage_mean_mean'), 0)} — o baseline não reduz variância e desloca as "
        'vantagens para cima, como na Q4. Ainda assim, lotes de 1.024 com alvo quase MC tornam cada atualização '
        f"muito informativa (450 com {both('num_steps-128', 'updates_to_450', 0)} atualizações, contra "
        f"{both('baseline', 'updates_to_450', 0)} no baseline): faltam atualizações (488), não sinal. A previsão "
        f"errou para o ator em n = 1, que aprendeu como o baseline (450 aos {both('num_steps-1', 'steps_to_450', fn=kilo)} "
        f"passos, contra {both('baseline', 'steps_to_450', fn=kilo)}): o erro de V é quase o mesmo para as duas ações "
        '(depende do tempo até o truncamento, que a ação não muda) e só desloca a vantagem.',
    ]
    caption = (CAPTION.replace('média por blocos de 5.000 passos', 'retorno: média por blocos de 5.000 passos; '
                               'value_loss e EV: mediana por blocos de 25.000 (o lote típico)') +
               ' Tabela: V̄ − Ḡ = viés do crítico final contra retornos Monte Carlo reais da própria política.')
    return section('Q2 · Horizonte do retorno de n passos (a2c.num_steps)',
                   'n = 1 (alvo r + γV(s′)) tem mais viés e menos variância; n = 128, quase Monte Carlo, o contrário. '
                   'Espero value_loss crescente com n e explained_variance alta, porém enganosa, com n = 1; o ator deve '
                   'sofrer nos extremos.',
                   'num_steps = 1, 5 (baseline), 32 e 128, duas seeds; lotes de 8, 40, 256 e 1.024; 62.500, 12.500, '
                   '1.953 e 488 atualizações. Depois do treino, cada crítico final foi comparado com os retornos Monte '
                   'Carlo da própria política (~28 mil transições; <i>scripts/critic_check_a2c.py</i>).',
                   [figure(OUT / 'q2.png'), P(caption, 'small')], tab, explanation, c)


def q3():
    c = ['ent_coef-0', 'baseline', 'ent_coef-0.1']
    names = {'ent_coef-0': 'ent_coef = 0', 'baseline': '0,01 (baseline)', 'ent_coef-0.1': 'ent_coef = 0,1'}
    side_w = CONTENT_W * 0.40
    rows = [['Config.', 'Entropia<br/>final', 'Retorno<br/>final', 'Avaliação<br/>(s1, s2)']]
    for name in c:
        a, b = f'{name}-s1', f'{name}-s2'
        rows.append([names[name].replace(' (baseline)', '<br/>(baseline)'),
                     pair(name, 'final_entropy', lambda x: num(x, 2)).replace(' / ', '<br/>'),
                     pair(name, 'final_episodic_return_mean_last100', lambda x: num(x, 0)).replace(' / ', '<br/>'),
                     f'{ev(a)}<br/>{ev(b)}'])
    tab = table(rows, cols(0.27, 0.2, 0.19, 0.34, width=side_w), ST)
    fig_w = CONTENT_W - side_w - 8
    visual = Table([[figure(OUT / 'q3.png', width=fig_w), [tab, Spacer(1, 4), P(
        CAPTION + ' Pontilhado: ln 2 (política uniforme). Retorno final = média dos últimos 10% do treino.',
        'small')]]], colWidths=[fig_w + 8, side_w])
    visual.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 0),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 0), ('TOPPADDING', (0, 0), (-1, -1), 0),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 0)]))
    explanation = [
        'O bônus −β·H impede o colapso prematuro: sem ele, um gradiente favorável por acaso torna a softmax '
        'determinística, ∇log π → 0 e a exploração acaba. Com β = 0,1 a entropia fica em '
        f"{both('ent_coef-0.1', 'final_entropy', 2)} do início ao fim e o retorno estaciona em "
        f"{both('ent_coef-0.1', 'final_episodic_return_mean_last100', 0)} (máx. "
        f"{both('ent_coef-0.1', 'max_return_last100', 0)}): o ótimo do objetivo regularizado é "
        'π(a|s) ∝ exp(A(s,a)/β), e onde a diferença de vantagem entre as ações é comparável a β a ação errada '
        'mantém probabilidade relevante — em 500 passos alguma dessas escolhas derruba a haste. Confirmado.',
        'Sem bônus, as duas seeds funcionaram (retorno final '
        f"{both('ent_coef-0', 'final_episodic_return_mean_last100', 1)}, o melhor da varredura) e a entropia caiu menos "
        f"do que previ ({both('ent_coef-0', 'final_entropy', 2)} contra {both('baseline', 'final_entropy', 2)}), sem "
        'colapsar. O CartPole é benevolente: recompensa densa, duas ações, política inicial quase uniforme e falhas '
        'rápidas que dão sinal em todo episódio; como na maioria dos estados as ações são quase equivalentes '
        '(vantagens ≈ 0), o gradiente só a torna determinística onde importa. Com mais ações ou recompensa rara, a '
        'política pode se fixar na primeira ação com vantagem positiva por acaso antes de achar a recompensa, e sem '
        'entropia nada a tira de lá (ver o Extra).',
    ]
    return section('Q3 · Coeficiente de entropia (a2c.ent_coef)',
                   'Sem bônus, entropia caindo mais cedo e mais perto de 0; no CartPole deve funcionar, com risco de uma '
                   'seed estagnar. Com 0,1, entropia perto de ln 2 e retorno parado abaixo de 500.',
                   'ent_coef = 0, 0,01 (baseline) e 0,1, duas seeds.', [visual], None, explanation, c)


def q4():
    c = ['baseline', 'no_baseline']
    names = {'baseline': 'com baseline', 'no_baseline': 'sem baseline'}
    rows = [['Configuração', 'advantage_<br/>mean (média)', 'advantage_<br/>std (mediana)', 'grad_norm<br/>(mediana)',
             'Entropia final', 'Entropia<br/>&lt; 0,1 em', 'Retorno final', 'Avaliação s1', 'Avaliação s2']]
    for name in c:
        a, b = f'{name}-s1', f'{name}-s2'
        rows.append([names[name], pair(name, 'advantage_mean_mean', lambda x: num(x, 1)),
                     pair(name, 'advantage_std_median', lambda x: num(x, 2)),
                     pair(name, 'grad_norm_median', lambda x: num(x, 1 if x > 1 else 2)),
                     pair(name, 'final_entropy', lambda x: num(x, 3)),
                     pair(name, 'steps_entropy_below_0.1', lambda x: kilo(x, 'k')),
                     pair(name, 'final_episodic_return_mean_last100', lambda x: num(x, 0)), ev(a), ev(b)])
    tab = table(rows, cols(0.115, 0.135, 0.135, 0.1, 0.115, 0.085, 0.075, 0.12, 0.12), ST)
    explanation = [
        'Subtrair b(s) não muda o gradiente esperado — a média de ∇log π(a|s) sob a ~ π é ∇Σπ(a|s) = ∇1 = 0 —, '
        'mas muda o segundo momento dos pesos: com baseline, média '
        f"{both('baseline', 'advantage_mean_mean', 1)} e desvio mediano {both('baseline', 'advantage_std_median', 2)}; "
        f"sem, média +{both('no_baseline', 'advantage_mean_mean', 0)} e desvio mediano "
        f"{both('no_baseline', 'advantage_std_median', 1)}. A norma do gradiente (mediana) sobe de "
        f"{both('baseline', 'grad_norm_median', 2)} para {both('no_baseline', 'grad_norm_median', 0)}, e "
        f"{both('no_baseline', 'frac_clipped', fn=lambda x: num(100 * x, 0))}% dos passos são cortados pelo "
        f"clipping (0,5), contra {both('baseline', 'frac_clipped', 2, fn=lambda x: num(100 * x, 0))}% com baseline: a "
        'direção passa a ser dominada pelo termo comum a todas as amostras. Confirmado.',
        'Com R &gt; 0 sempre, todo passo aumenta log π da ação amostrada, seja ela boa ou ruim; a informação útil '
        'está só na variação de R em torno da média, pequena perto dela. A ação mais provável é amostrada — e '
        'reforçada — mais vezes, um ciclo que torna a política determinística: entropia &lt; 0,1 aos '
        f"{' e '.join(kilo(v(f'no_baseline-s{s}', 'steps_entropy_below_0.1')) for s in (1, 2))} passos e "
        f"{both('no_baseline', 'final_entropy', 3)} no fim (baseline: {both('baseline', 'final_entropy', 2)}); o bônus "
        'de 0,01 é irrelevante diante de pesos ~50. Sem exploração não há correção: a seed 2 chegou a 500 e caiu para '
        f"{num(v('no_baseline-s2', 'final_episodic_return_mean_last100'), 0)}; a seed 1 chegou a "
        f"{num(v('no_baseline-s1', 'max_return_last100'), 0)} e terminou em "
        f"{num(v('no_baseline-s1', 'final_episodic_return_mean_last100'), 0)}.",
    ]
    return section('Q4 · Ablação do baseline (a2c.use_baseline = false)',
                   'Sem baseline o peso é R, sempre positivo (dezenas): advantage_mean ≫ 0 e advantage_std várias vezes '
                   'maior; como todo peso é positivo, a entropia deve colapsar mais cedo e o retorno ficar mais lento e '
                   'instável, embora o gradiente esperado seja o mesmo.',
                   'use_baseline = true (baseline) e false, duas seeds.',
                   [figure(OUT / 'q4.png'), P('Convenções como na Q1 (média por blocos de 5.000 passos; contínua = '
                                              'seed 1, tracejada = seed 2).', 'small')], tab, explanation, c)


def extra():
    c = ['lunar-cartpole_cfg', 'lunar-cartpole_cfg-3M', 'lunar-e16n16-3M', 'lunar-g995-3M', 'lunar-g995-ent001-3M',
         'lunar-g999-ent001-3M', 'lunar-g999-lr25-e16-5M', 'lunar-g999-e64-5M', 'lunar-g995-lr25-e16-5M']
    in_table = ['lunar-g995-ent001-3M', 'lunar-g999-ent001-3M', 'lunar-g999-lr25-e16-5M', 'lunar-g999-e64-5M',
                'lunar-g995-lr25-e16-5M']
    names = {'lunar-g995-ent001-3M': 'γ 0,995, ent_coef 0,001', 'lunar-g999-ent001-3M': 'γ 0,999, ent_coef 0,001',
             'lunar-g999-lr25-e16-5M': 'γ 0,999 + LR 2,5e-4', 'lunar-g999-e64-5M': 'γ 0,999 + 64 amb.',
             'lunar-g995-lr25-e16-5M': 'γ 0,995 + LR 2,5e-4'}
    rows = [['Configuração', 'Passos', 'Máx. (100 ep.)', 'Retorno final', 'Duração final',
             'EV mediana', 'Avaliação s1', 'Avaliação s2']]
    for name in in_table:
        seeds = [s for s in (1, 2) if 'eval_mean' in S.get(f'{name}-s{s}', {})]
        if not seeds:
            continue
        get = lambda key, nd: ' / '.join(num(v(f'{name}-s{s}', key), nd) for s in seeds)
        steps = v(f'{name}-s{seeds[0]}', 'num_iterations') * v(f'{name}-s{seeds[0]}', 'batch_size')
        rows.append([names[name], f'{num(steps / 1e6, 1)}M', get('max_return_last100', 0),
                     get('final_episodic_return_mean_last100', 0), get('final_episodic_length_mean_last100', 0),
                     get('ev_median', 2), ev(f'{name}-s1'), ev(f'{name}-s2') if 2 in seeds else '—'])
    tab = table(rows, cols(0.22, 0.07, 0.11, 0.11, 0.105, 0.085, 0.15, 0.15), ST)

    m = lambda label, key='max_return_last100': num(v(label, key), 0)
    fin = lambda label: m(label, 'final_episodic_return_mean_last100')
    best, best2 = 'lunar-g995-ent001-3M-s1', 'lunar-g995-ent001-3M-s2'
    explanation = [
        'Previsão certa quanto ao orçamento e errada quanto ao crítico. A config do CartPole aprende a não cair (de '
        f"−200 a ~+{m('lunar-cartpole_cfg-s1')} em 400 mil passos) e passa a pairar: episódios de "
        f"{m('lunar-cartpole_cfg-s1', 'final_episodic_length_mean_last100')} passos, retorno final "
        f"{fin('lunar-cartpole_cfg-s1')} (com 3M: máx. {m('lunar-cartpole_cfg-3M-s1')}, final "
        f"{fin('lunar-cartpole_cfg-3M-s1')}; avaliação {ev('lunar-cartpole_cfg-3M-s1')}). Em relação ao CartPole: (i) o "
        'bônus de pouso (+100) vem centenas de passos depois e cada passo com motor custa, então γ maior valoriza '
        'pousar (a 200 passos, 0,99<super>200</super> ≈ 13% e 0,995<super>200</super> ≈ 37%); (ii) com 4 ações e pouso '
        'preciso, ent_coef = 0,01 mantém a entropia em ~0,6 e aciona motores ao acaso — γ = 0,995 sozinho chegou a '
        f"{m('lunar-g995-3M-s1')}, e com ent_coef = 0,001 a {m(best)}; (iii) são precisos 3–5M passos. O crítico não é "
        'o gargalo: a EV passa de 0,9 cedo, pois a recompensa de modelagem torna V previsível pelo estado (no CartPole, '
        'o tempo até o truncamento é invisível).',
        'O limite é o ator. Todas as variantes oscilam: sobem (máx. de 79 a 225) e recaem primeiro no ótimo local de '
        'pairar — a duração salta para ~900–1.000 passos (truncamento) — e, em alguns runs, depois para voos que terminam '
        f"em queda (retorno final de −110 a −150). Nem passos menores (LR 2,5e-4: final {fin('lunar-g999-lr25-e16-5M-s1')} "
        f"e {fin('lunar-g999-lr25-e16-5M-s2')} com γ = 0,999; {fin('lunar-g995-lr25-e16-5M-s1')} e "
        f"{fin('lunar-g995-lr25-e16-5M-s2')} com γ = 0,995) nem lotes maiores (64 amb.: {fin('lunar-g999-e64-5M-s1')} e "
        f"{fin('lunar-g999-e64-5M-s2')}) evitaram a recaída. O A2C não limita quanto a política muda por atualização "
        '(não há região de confiança, como no PPO) e é on-policy: pairando, o agente quase não vê pousos e perde o sinal '
        'que o traria de volta; um ent_coef maior ajudaria a sair dali, mas atrapalha a precisão do pouso. Cheguei perto '
        f"sem resolver: γ = 0,995 com ent_coef = 0,001 passou de 200 na seed 1 (máx. {m(best)}; avaliação {ev(best)}) e "
        f"chegou a {m(best2)} na seed 2 (avaliação {ev(best2)}).",
    ]
    caption = ('Média por blocos de 25.000 passos (EV: mediana); pontilhados: 200 (resolvido), 1.000 passos '
               '(truncamento) e ln 4. “+”: sobre γ e ent_coef = 0,001, com 16 ambientes (LR) ou LR 7e-4 (64 amb.). '
               'Tabela: valores finais = últimos 10% do treino (s1 / s2).')
    return section('Extra · LunarLander-v3',
                   'Com a config do CartPole e 500 mil passos, não deve resolver: recompensas de −100/+100 no fim de '
                   'episódios de até 1.000 passos exigem mais passos, retornos mais longos e entropia menor no fim. '
                   'Espero value_loss muito maior que no CartPole e explained_variance baixa no início.',
                   'Uma seed a partir da config do CartPole (500 mil e 3M passos; 16 amb. × n = 16; γ = 0,995; '
                   'γ = 0,995 e 0,999 com ent_coef = 0,001); depois, duas seeds para a melhor delas e para três '
                   'correções com 5M passos (LR 2,5e-4 com γ = 0,999 e 0,995; 64 ambientes).',
                   [figure(OUT / 'lunar.png'), P(caption, 'small')], tab, explanation, c)


def limits():
    return [P('<b>Limites.</b> Duas seeds não bastam para conclusões universais (o baseline diverge na avaliação: '
              f"{ev('baseline-s1')} e {ev('baseline-s2')}); a mesma seed reproduz o treino bit a bit; SPS comparável "
              'só na Q1. Previsões, adições durante a execução e scripts: <i>EXPERIMENTOS_A2C.md</i>.', 'small')]


def main(extra=None):
    story = front() + q1() + q2() + q3() + q4() + (extra() if extra else [])
    build(PDF, story, header=f'{AUTHOR} | Aprendizado por Reforço | A2C',
          title=f'A2C no CartPole - {AUTHOR}', author=AUTHOR, subject='Atividade: Policy Gradient com A2C',
          keywords='A2C, actor-critic, policy gradient, CartPole, LunarLander')
    import pypdf
    print(f'{PDF} — {len(pypdf.PdfReader(str(PDF)).pages)} páginas')


if __name__ == '__main__':
    main(extra)
