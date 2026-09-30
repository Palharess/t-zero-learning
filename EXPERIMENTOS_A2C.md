# A2C — plano experimental registrado antes dos treinos

Autor: Gabriel Palhares Siqueira (individual) · palhares@discente.ufg.br

Ambiente: CartPole-v1. Baseline: `configs/a2c_cartpole.yml` sem alterações
(8 ambientes, `num_steps=5`, `ent_coef=0,01`, `use_baseline=true`,
LR 7e-4, 500.000 passos). Uma alteração por vez, seeds 1 e 2 em todas as
configurações. `capture_video=false` apenas para não renderizar; a avaliação
final de 10 episódios (política estocástica, como no harness) continua ativa.
O baseline é reutilizado nas quatro questões.

As previsões abaixo foram escritas antes de qualquer treino da varredura
(só um teste de 40 mil passos do baseline, para validar o pipeline, foi rodado
e descartado). Elas não são resultados.

## Q1 — número de atores (`num_envs`)
Valores: 1, 8 (baseline), 64. Lote = `num_envs × 5`: 5, 40 e 320 transições;
atualizações em 500 mil passos: 100.000, 12.500 e 1.562.

Previsão: com 1 ambiente, cada gradiente usa 5 transições consecutivas do mesmo
episódio; espero `policy_loss` muito ruidoso (picos e troca de sinal a cada lote)
e retorno instável, mesmo com 8× mais atualizações. Com 64 ambientes o
`policy_loss` deve ficar suave e perto de zero, mas com só 1.562 atualizações
o retorno deve subir mais devagar por passo de ambiente. O SPS deve crescer com
`num_envs` (inferência em lote, uma atualização a cada 320 passos), mas de forma
sublinear: 12 threads e a comunicação do `AsyncVectorEnv` saturam.

Gráficos: `charts/episodic_return_mean_last100`, `losses/policy_loss`,
`charts/SPS`; além disso, retorno por número de atualizações e por tempo de
relógio, para separar o efeito do tamanho do lote do efeito do número de
atualizações. Os runs de Q1 são executados um de cada vez, sem outro treino em
paralelo, para que o SPS seja comparável.

## Q2 — horizonte do retorno (`a2c.num_steps`)
Valores: 1, 5 (baseline), 32, 128. Lote = 8, 40, 256 e 1.024 transições;
atualizações: 62.500, 12.500, 1.953 e 488.

Previsão: `num_steps=1` (alvo r + γV(s')) tem mais viés e menos variância;
`num_steps=128` é quase Monte Carlo, com menos viés e mais variância. Espero
`value_loss` crescente com n e `explained_variance` alta porém enganosa com n=1,
porque o alvo contém a própria previsão V(s'). O ator deve aprender devagar com
n=1 (vantagem enviesada, lote de 8) e com n=128 (poucas atualizações e
vantagens ruidosas); 5–32 devem ficar no meio.

Gráficos: `charts/episodic_return_mean_last100`, `losses/value_loss`,
`losses/explained_variance`.

## Q3 — coeficiente de entropia (`a2c.ent_coef`)
Valores: 0, 0,01 (baseline), 0,1.

Previsão: sem bônus, a entropia deve cair mais cedo e mais perto de 0; no
CartPole isso provavelmente ainda funciona (2 ações, recompensa densa), mas com
mais risco de uma seed oscilar ou estagnar. Com 0,1 espero entropia alta (perto
de ln 2 ≈ 0,69) e retorno parado abaixo de 500, porque o termo de entropia
compete com a vantagem e mantém ações aleatórias em estados críticos.

Gráficos: `charts/episodic_return_mean_last100`, `losses/entropy`.

## Q4 — ablação do baseline (`a2c.use_baseline=false`)
Valores: `true` (baseline) e `false`.

Previsão: sem baseline o peso é R (sempre positivo, dezenas no CartPole):
espero `advantage_mean` muito acima de 0 e `advantage_std` várias vezes maior
que com baseline, cuja média fica perto de 0. Como todo peso é positivo, toda
ação amostrada tem a log-probabilidade empurrada para cima — a mais provável é
amostrada e reforçada mais vezes —, então espero a entropia colapsar mais cedo e
o retorno subir mais devagar e de forma mais instável, embora o gradiente
esperado seja o mesmo.

Gráficos: `charts/episodic_return_mean_last100`, `charts/advantage_std`,
`charts/advantage_mean`, `losses/entropy`.

## Extra — LunarLander-v3
Ponto de partida: a config do CartPole com `env_id=LunarLander-v3`.

Previsão: com a config do CartPole e 500 mil passos o agente não deve resolver.
Recompensas de −100/+100 no fim do episódio e episódios de até 1.000 passos
exigem mais passos de treino, retornos de n passos mais longos para propagar o
termo terminal e entropia menor no fim (4 ações, pouso preciso). Espero
`value_loss` muito maior que no CartPole e `explained_variance` baixa no início.

### Adições durante a execução (registradas quando foram decididas)
- Q1: depois dos seis runs de Q1, acrescentei um controle com 64 ambientes e
  4 milhões de passos (12.500 atualizações, as mesmas do baseline), para
  separar o efeito do tamanho do lote do efeito do número de atualizações.
- Q2: depois dos treinos, `scripts/critic_check_a2c.py` compara cada crítico
  final com os retornos Monte Carlo da própria política (a armadilha da
  `explained_variance` com n=1).
- Extra: exploração com uma seed. A config do CartPole (500 mil passos) subiu
  até ~+70 e depois passou a pairar (episódios de ~900 passos, retorno ~0);
  16 ambientes × n=16 ficou em ~−110 com 1,4M passos. A variante 32 × 8 foi
  cancelada antes de começar. Hipótese nova: γ=0,99 desconta demais o bônus de
  pouso (centenas de passos à frente) e a entropia alta atrapalha a precisão;
  três variantes com 3M passos: γ=0,995; γ=0,995 com `ent_coef=0,001`;
  γ=0,999 com `ent_coef=0,001`.
  A melhor configuração será repetida com duas seeds.
- Extra, segunda rodada: nas quatro variantes de 3M o retorno sobe até
  +50…+100 em 250–500 mil passos e depois colapsa e oscila (máximos de 140 a
  185), com o crítico excelente (EV 0,97–1,0). Hipótese: o limite é o tamanho
  e o ruído do passo do ator (LR constante 7e-4, lote 40; o SB3 usa LR com
  decaimento). Duas correções, cada uma com duas seeds e 5M passos, ambas com
  γ=0,999 e `ent_coef=0,001`: passos menores (LR 2,5e-4, 16 ambientes) e
  direções menos ruidosas (64 ambientes, lote 320, LR 7e-4).
- Extra, terceira rodada: as duas correções também oscilaram (LR 2,5e-4 chegou
  a ~200 em 1M passos e caiu; 64 ambientes colapsaram). Segunda seed da melhor
  configuração da exploração (γ=0,995, `ent_coef=0,001`, 3M passos) para medir
  a robustez do resultado de ~225. Por último, a combinação dessa configuração
  com os passos menores (LR 2,5e-4, 16 ambientes, 5M passos, duas seeds).

## Integridade dos resultados
Registrar IDs dos runs, overrides, seeds, versões e resultados individuais;
não selecionar seeds favoráveis. As conclusões só serão escritas depois dos
treinos, a partir das métricas registradas.

## Execução e reprodução

Implementação (Partes 1–3): `compute_n_step_returns` percorre o rollout de trás
para frente com `R = r_t + γ(1 − d_t)R`, partindo de `V(s_T)`, coluna a coluna;
`compute_policy_loss` devolve `−mean(logπ · A)` com `A = R − V` (ou `R`)
destacado do grafo (`detach`), para que o gradiente do ator não chegue ao
crítico; `get_action_and_value` usa `Categorical(logits=...)`, amostra ou toma o
argmax, e avalia `log_prob`/`entropy` da ação recebida.

```bash
python -m pytest tests/test_a2c.py                        # 14 testes
python scripts/sweep_a2c.py --jobs 1 --only baseline num_envs-1 num_envs-64   # Q1, um run por vez
python scripts/sweep_a2c.py --jobs 4 --only num_steps-1 num_steps-32 num_steps-128 ent_coef-0 ent_coef-0.1 no_baseline
python scripts/sweep_a2c.py --jobs 1 --only num_envs-64-4M                     # controle da Q1
python scripts/sweep_a2c.py --jobs 3 --seeds 1 --only lunar-cartpole_cfg lunar-cartpole_cfg-3M lunar-e16n16-3M lunar-g995-3M lunar-g995-ent001-3M lunar-g999-ent001-3M
python scripts/sweep_a2c.py --jobs 4 --only lunar-g999-lr25-e16-5M lunar-g999-e64-5M
python scripts/sweep_a2c.py --jobs 1 --seeds 2 --only lunar-g995-ent001-3M
python scripts/sweep_a2c.py --jobs 2 --only lunar-g995-lr25-e16-5M
python scripts/critic_check_a2c.py num_steps-1-s1 num_steps-1-s2 baseline-s1 baseline-s2 num_steps-32-s1 num_steps-32-s2 num_steps-128-s1 num_steps-128-s2
python scripts/analyze_a2c.py        # resumos e figuras (numpy + matplotlib)
python scripts/build_a2c_report.py   # PDF (reportlab)
python scripts/wandb_report_a2c.py   # relatório no W&B (wandb-workspaces)
```

Cada run passa pelo harness original via `scripts/run_a2c.py`: o nome do run no
W&B é o rótulo `<config>-s<seed>`, o grupo é a configuração e as tags são as
questões; tudo o que vai para o W&B também é gravado em
`runs/a2c/<rótulo>/metrics.jsonl`, com o ID do run em `metadata.json`.
`capture_video=false`; `OMP_NUM_THREADS=1` em todos os treinos. Os runs de Q1
rodaram sozinhos na máquina; os demais, até seis ao mesmo tempo (o SPS deles
não é comparável). Com a mesma seed o treino é determinístico: o controle da
Q1 reproduz bit a bit os primeiros 500 mil passos do run de 64 ambientes.

Ambiente: Windows 11, Ryzen 5 5600X (6 núcleos/12 threads), 32 GB, sem GPU;
Python 3.10 (uv) com `torch==2.9.1+cpu` e os demais pinos de
`requirements.txt` (lista completa em `reports/a2c/dependency-versions.txt`).
Diferente do DQN, o `wandb==0.21.1` fixado aceitou a chave nova quando passada
por `WANDB_API_KEY` no `.env`, e os runs foram registrados online. O
LunarLander usa `box2d-py==2.3.5` compilado com o MSVC (`vcvars64.bat`,
`DISTUTILS_USE_SDK=1`, `pip install --no-build-isolation box2d-py==2.3.5`),
como no Dockerfile do curso.

## Resultados (resumo; detalhes no relatório)

- Q1: 8 ambientes foi o melhor por passo e por tempo (450 aos 117/148 mil
  passos, ~20 s); 1 ambiente não chegou a 450 (policy_loss ≈ 0 em 94% das
  atualizações, com picos); 64 ambientes são os melhores por atualização
  (450 com 1.563/1.887 atualizações no controle), mas só fazem 1.562
  atualizações em 500 mil passos. SPS ≈ 1.000 / 6.200 / 18.100.
- Q2: value_loss típico cresce com n e a EV cai de ~0,95 (n=1) para ~0
  (n=128); contra retornos Monte Carlo, o crítico de n=1 prevê V ≈ 100 para
  um retorno real de ~80 (viés +20), o de n=32 é o mais calibrado e o de
  n=128 subestima ~43. n=1 aprendeu tão rápido quanto o baseline.
- Q3: ent_coef=0 funcionou (retorno final ~498 nas duas seeds) com entropia 0,44–0,47; 0,1 travou a
  entropia em ~0,60 e o retorno em ~350.
- Q4: sem baseline, pesos com média +51/+72 e DP mediano centenas de vezes maior; 99–100%
  dos passos cortados pelo clipping; entropia < 0,1 aos 86/107 mil passos.
- Extra: melhor configuração γ=0,995 e ent_coef=0,001 (3M passos): máx.
  225/195, avaliação 227,6 ± 66,0 e 178,1 ± 116,6 — perto de 200, sem se
  sustentar. Todas as variantes oscilam e recaem no ótimo local de pairar.

Dois testes fora de `tests/test_a2c.py` falham neste Windows por motivos
anteriores a esta atividade: `test_config_reference` (o gerador escreve
caminhos com `\`) e os smoke tests de `dqn` (sem entrada em
`tests/helpers.py::SMOKE_SETTINGS` no upstream). Os smoke tests de treino e
retomada do A2C passam.
