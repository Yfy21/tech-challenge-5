# Datathon Passos Mágicos — risco de defasagem escolar

Projeto da Fase 5 (Deep Learning and Unstructured Data) da Pós-Tech em Data Analytics da FIAP. O objeto é
a base PEDE da Associação Passos Mágicos (avaliações de 2022, 2023 e 2024), e a entrega tem três frentes:
responder às 11 perguntas analíticas propostas no datathon, treinar um modelo preditivo de risco de
defasagem escolar e disponibilizar tudo em um dashboard Streamlit.

A defasagem (`defas = fase − fase_ideal`) é o eixo do projeto porque é o indicador acionável: um aluno
defasado hoje já é visível a olho nu; o que a associação não tem é um instrumento que aponte **quem vai se
defasar no ano que vem** — em especial quem está em dia hoje. O modelo foi desenhado exatamente para esse
alerta precoce (desenho t→t+1, detalhado abaixo).

## Estrutura do repositório

```
data/raw/               dataset.xlsx original (3 abas: PEDE2022, PEDE2023, PEDE2024)
data/processed/         pede_unificado.csv (painel), evasao.csv, base_risco.csv
scripts/01_clean.py     harmonização das 3 planilhas em painel longo
scripts/02_transform.py correções semânticas e marcação de em_risco
scripts/03_analysis.py  agregados que sustentam as respostas às 11 perguntas
scripts/04_features.py  base de features t→t+1 do modelo (base_risco.csv)
notebooks/01_eda.ipynb  análise exploratória — respostas às 11 perguntas
notebooks/02_modelagem.ipynb  modelos, comparação, validações e exportação
models/                 artefatos exportados (risk_rf.pkl, risk_nn.keras, metadata)
dashboard/app.py        aplicação Streamlit em 3 seções
outputs/                CSVs agregados consumidos pela EDA
```

## 1. Limpeza (`scripts/01_clean.py`)

As três planilhas têm estruturas de colunas próprias — nomes, escalas e até a presença de indicadores
variam de ano para ano — então o primeiro passo harmoniza tudo em um único painel longo, com uma linha por
(id_aluno, ano): 3.030 linhas e 1.661 alunos. As decisões de harmonização que mais importam:

- Os indicadores foram todos levados à escala 0–10, não à 0–1.
- O IPP ausente em 2022 é reconstruído algebricamente a partir do INDE (cujos pesos são conhecidos), em
  vez de imputado estatisticamente.
- `Atingiu PV` e `Defas` são recriados a partir de IPV/Fase — não se confia no dado bruto, pois ambos
  estão sinalizados como propensos a erro nos documentos-fonte.
- `Turma`, `Nome` e `Cg` são descartados por serem redundantes ou ruidosos; `INDE 23`/`Pedra 23` (colunas
  remanescentes quase vazias na planilha 2023) e `Ativo/Inativo` (constante em 2024) também caem.
- A evasão é calculada por diferença de conjuntos de `id_aluno` entre anos (presente no ano N, ausente no
  N+1), gerando `evasao.csv` — não se usa a coluna `Ativo/Inativo`, que é constante.

## 2. Transformação e análise (`scripts/02_transform.py`, `03_analysis.py`, `notebooks/01_eda.ipynb`)

O segundo passo aplica correções semânticas sobre o painel e marca `em_risco` (defas < 0), tomando o
cuidado de não confundi-lo com `atingiu_pv` — são alvos distintos. O terceiro exporta para `outputs/` os
agregados que sustentam as respostas às 11 perguntas; o notebook `01_eda.ipynb` consome esses CSVs e
responde as perguntas uma a uma. Os achados que ancoram o resto do projeto:

- A defasagem cai ano a ano no agregado (69,9% → 54,4% → 47,8%) e cai ainda mais forte na coorte fixa de
  alunos presentes nos três anos (67,3% → 34,8%) — evidência de efetividade do acompanhamento.
- A evasão fica em 30,2% (2022→2023) e 24,6% (2023→2024). A rigor a métrica mede "não retorno à avaliação
  PEDE seguinte", e o rótulo importa: parte dos ausentes pode ter saído por conclusão ou motivo externo,
  não por abandono.
- Os sinais univariados mais fortes contra o risco futuro são IPP, a própria defasagem atual, IPV e INDE —
  o que depois se confirma na importância de features do modelo.

## 3. Features e modelagem (`scripts/04_features.py`, `notebooks/02_modelagem.ipynb`)

A base do modelo (`base_risco.csv`, 1.290 pares) segue o desenho **t→t+1**: cada linha é um par
aluno × anos consecutivos, com as features observadas no ano t e o alvo `em_risco` observado em t+1. Não
há circularidade — defas e indicadores do ano t são preditores legítimos do risco futuro; o que não pode é
usar os do próprio ano-alvo. Entram 14 features numéricas (indicadores, fase, idade, tempo de vínculo e as
tendências d_inde/d_ieg/d_ida, que são NaN quando não existe ano anterior) e 2 categóricas (gênero, tipo
de instituição). O IAN fica de fora por ser função determinística de defas — seria redundante.

O split de teste separa 20% dos **alunos** (GroupShuffleSplit por `id_aluno`), não 20% das linhas, porque
o mesmo aluno em anos diferentes nos dois lados do split inflaria as métricas. Imputação e escala são
ajustadas só no treino. Três modelos disputam:

| Modelo | Acurácia | F1 | ROC-AUC | Recall novos defasados* |
|---|---|---|---|---|
| Regressão Logística (baseline) | 0,736 | 0,742 | 0,842 | 0,27 |
| **RandomForest (campeão)** | **0,782** | **0,797** | **0,871** | **0,73** |
| Rede Neural (Keras) | 0,751 | 0,760 | 0,850 | 0,55 |

\* recall no subgrupo crítico do teste: alunos em dia no ano t que se defasaram em t+1 (22 casos) —
exatamente quem um alerta precoce existe para pegar.

O RandomForest vence em todas as métricas agregadas e, sobretudo, no subgrupo crítico: pega 73% dos alunos
que vão se defasar estando hoje em dia, contra 27% do baseline linear. A rede neural (MLP com duas camadas
ocultas, dropout e early stopping por AUC em validação separada por aluno) fica próxima e cumpre o
requisito de Deep Learning da fase; com ~1.300 exemplos tabulares, o teto de uma MLP contra árvores é
esperado.

## 4. Validações — sem pressupor que os números estão certos

**Evasão.** A taxa de 20–30% parecia alta, então foi checada contra três hipóteses alternativas antes de
ser aceita: não é artefato de formatura (a fase 8, a última, tem a **menor** evasão, 6,3%; o pico está na
fase 6, 43,1%, e nas idades 14–17, 35,0%); as saídas são permanentes (só 4 dos 260 evadidos de 2022→2023
reaparecem em 2024, 1,5%); e os ids são inteiros sequenciais estáveis, sem sinal de recodificação entre
anos. O número se sustenta — com o caveat de rótulo já citado.

**Baseline de persistência.** A regra "quem está defasado hoje continua defasado no ano que vem" não exige
modelo algum e já acerta bastante (acurácia 0,686; ROC-AUC 0,744 usando −defas como score). O campeão
supera com folga (0,782 e 0,871) — o pipeline agrega sinal real além do óbvio.

**Validação temporal.** O split por aluno impede memorizar alunos, mas não impede aprender padrões
específicos de um ano; treinar só nos pares 2022→2023 e testar nos pares 2023→2024 reproduz o cenário real
de uso. O poder de **ordenação** se mantém quase intacto (ROC-AUC 0,862 contra 0,871 do split aleatório),
mas o limiar fixo de 0,5 descalibra (recall 0,56), porque a prevalência do alvo despenca entre treino e
teste (61% → 44%) e o treino de 2022 não enxerga as tendências d_* (não existe 2021). A conclusão
operacional: o modelo deve ser usado como **ranking de priorização** — atender os top-N alunos de maior
probabilidade conforme a capacidade da equipe — em vez de corte fixo; o limiar ajustável do dashboard
existe para isso.

## 5. Dashboard (`dashboard/app.py`)

Aplicação Streamlit em três seções: **Visão geral** (evolução da defasagem, evasão e indicadores por ano),
**Análise por fase** (heatmap de taxa de risco por fase × ano e recortes) e **Preditor de risco** (carrega
`risk_rf.pkl` e pontua um aluno a partir dos indicadores informados, com limiar ajustável). O tema segue a
identidade visual definida em `.streamlit/config.toml`.

## Como executar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pipeline de dados (a ordem importa — cada script espera a saída do anterior)
python scripts/01_clean.py
python scripts/02_transform.py
python scripts/03_analysis.py
python scripts/04_features.py

# notebooks (EDA e modelagem; o segundo exporta os artefatos de models/)
jupyter notebook notebooks/

# dashboard
streamlit run dashboard/app.py
```

Os artefatos de `models/` já estão versionados, então o dashboard roda sem re-treinar nada; re-executar
`02_modelagem.ipynb` regenera os mesmos arquivos (seeds fixas, SEED=42).
