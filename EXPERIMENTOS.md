# Plano experimental — registrado antes dos treinos

Ambiente: CartPole-v1. Baseline: configs/dqn_cartpole.yml, 500.000 passos.
Comparações com uma alteração por vez e seeds 1 e 2. Avaliação greedy final:
10 episódios. Desabilitar vídeo apenas para reduzir custo de renderização.

## Q1 — frequência da target network
Valores: 1, 500 (baseline), 10.000.
Previsão: com frequência 1, o alvo acompanha imediatamente a rede online,
aumentando o acoplamento do bootstrap e o risco de instabilidade de Q e retorno;
TD loss pequeno não garante uma política boa. Com 10.000, o alvo fica estável
por mais tempo, mas a propagação de valores é lenta e pode mudar em degraus.
Gráficos: retorno médio dos últimos 100 episódios, TD loss e Q médio.

## Q2 — replay buffer
Valores: 128, 10.000 (baseline), 100.000.
Previsão: o buffer de 128 terá baixa diversidade e transições temporalmente
próximas, além de esquecer rapidamente experiências antigas; isso pode
instabilizar Q e reduzir retorno. Um buffer de 100.000 conserva diversidade,
mas dados antigos podem retardar a adaptação à distribuição atual.
Gráficos: retorno médio dos últimos 100 episódios e Q médio.

## Q3 — taxa de aprendizado (extra)
Valores: 0,000025, 0,00025 (baseline), 0,0025.
Previsão: a taxa pequena deve reduzir oscilações, mas aprender mais devagar;
a grande pode produzir atualizações excessivas, picos de TD loss e instabilidade
de Q, prejudicando retorno. Reportar retorno, TD loss e Q para distinguir
lentidão de aprendizado de instabilidade das estimativas.

## Integridade dos resultados
As previsões acima não são resultados. Conclusões e gráficos só serão
preenchidos depois das execuções. Registrar IDs dos runs, configurações,
seeds, versões de dependências e resultados individuais. Reutilizar o baseline
nas três comparações; não selecionar apenas as seeds favoráveis.

## Execução e reprodução

Autor: Gabriel Palhares Siqueira (individual).

Os três blocos foram implementados em `algorithms/dqn.py`.
O alvo é `r + gamma * (1 - done) * max_a Q_target(s_next, a)`.
O `done` usado pelo harness corresponde a término verdadeiro; o limite de
500 passos é truncamento e não remove o bootstrap. A amostragem usa reposição,
como exigido pelo contrato do buffer, e os alvos são calculados sem gradientes.

```bash
python -m pytest tests/test_dqn.py
WANDB_MODE=offline python scripts/run_assignment.py --label baseline-s1 --seed 1
WANDB_MODE=offline python scripts/sweep_assignment.py
```

O segundo comando executa o baseline com seed 1; o terceiro executa as outras
13 combinações. Cada treino mantém os 500.000 passos da configuração original.
`capture_video=false` evita renderização; a avaliação greedy de 10 episódios e
o salvamento do modelo continuam ativos. PyTorch usa uma thread por processo,
apropriada para esta rede pequena; no máximo três treinos da varredura são
executados simultaneamente. O algoritmo e os parâmetros restantes são iguais
aos do baseline. `dependency-versions.txt` registra as versões usadas.

Os scripts preservam os arquivos originais do W&B e uma cópia JSONL das métricas,
com labels explícitos e IDs em `runs/assignment/<label>/metadata.json`.
A média e o desvio da avaliação resumem episódios dentro de uma seed; não são
intervalos de confiança nem estimativas da variabilidade entre seeds.
As seeds configuradas controlam Python, NumPy, PyTorch e o reset do ambiente
de treino. O harness original não chama `single_action_space.seed` para as
ações exploratórias nem passa uma seed explícita ao reset da avaliação.
Assim, repetir o mesmo valor de `seed` não garante reprodução bit a bit das
curvas. Mantivemos esse comportamento original em todas as configurações;
as duas execuções por configuração medem variabilidade, não pares idênticos
de trajetórias. Os dados registrados são a referência dos resultados.

Após configurar o login, os dados offline podem ser sincronizados:

```bash
wandb login
wandb sync --sync-all
```

Não confundir hipóteses previstas com observações; consultar as métricas reais
antes de redigir as conclusões. O nome do run identifica configuração e seed.

### Compatibilidade do login W&B

O `wandb==0.21.1` fixado em `requirements.txt` rejeita o formato novo de chave
no login interativo (erro de comprimento de 40 caracteres). Nesta execução,
o treino usou a versão original em modo offline, e a sincronização usou
`wandb==0.30.0` em outro ambiente virtual. Para repetir essa separação:

```bash
python -m venv .venv-sync
.venv-sync/bin/python -m pip install 'wandb>=0.22.3'
.venv-sync/bin/wandb login
.venv-sync/bin/wandb sync --sync-all
```

Referência: https://docs.wandb.ai/platform/app/settings-page/user-settings

### Validação

- `python -m pytest tests/test_dqn.py`: 7 testes passaram.
- `python -m pytest tests/test_dqn.py tests/test_agent_network.py tests/test_configs_load.py tests/test_overrides.py`: 32 testes passaram.
- Baseline: média greedy 500, desvio padrão 0, nas duas seeds (10 episódios cada).
