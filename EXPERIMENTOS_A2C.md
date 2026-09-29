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

## Integridade dos resultados
Registrar IDs dos runs, overrides, seeds, versões e resultados individuais;
não selecionar seeds favoráveis. As conclusões só serão escritas depois dos
treinos, a partir das métricas registradas.
