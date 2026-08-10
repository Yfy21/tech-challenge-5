# Datathon Passos Mágicos — risco de defasagem escolar

Projeto da Fase 5 (Deep Learning and Unstructured Data) da Pós-Tech em Data Analytics da FIAP. O objeto é
a base PEDE da Associação Passos Mágicos (avaliações de 2022, 2023 e 2024), e a entrega tem três frentes:
responder às 11 perguntas analíticas propostas no datathon, treinar um modelo preditivo de risco de
defasagem escolar e disponibilizar tudo em um dashboard Streamlit.

A defasagem (`defasagem = fase − fase_ideal`) é o eixo do projeto porque é o indicador acionável: um aluno
defasado hoje já é visível a olho nu; o que a associação não tem é um instrumento que aponte **quem vai se
defasar no ano que vem** — em especial quem está em dia hoje. O modelo foi desenhado exatamente para esse
alerta precoce (desenho t→t+1, detalhado abaixo).

## Estrutura do repositório

```
data/raw/               dataset.xlsx original (3 abas: PEDE2022, PEDE2023, PEDE2024)
data/processed/         pede_unificado.csv (painel), evasao.csv, base_risco.csv
scripts/01_clean.py     harmonização das 3 planilhas em painel longo + marcação de defasado
notebooks/02_eda.ipynb  análise exploratória — respostas às 11 perguntas
scripts/03_features.py  base de features t→t+1 do modelo (base_risco.csv)
notebooks/04_modelagem.ipynb  modelos, comparação, validações e exportação
models/                 artefatos exportados (risk_rf.pkl, risk_nn.keras, metadata)
dashboard/app.py        aplicação Streamlit em 3 seções
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
- `Gênero` é normalizado para os rótulos legíveis `Masc`/`Fem` — as planilhas alternam entre
  "Menino/Menina" e "Masculino/Feminino"; qualquer codificação numérica fica a cargo da modelagem.
- O IEG dos universitários (fase 8, 2024) traz zeros como placeholder — nenhum indicador é de fato
  avaliado para esses alunos —, então esses zeros viram NaN.
- Os zeros do IAA **não são notas**: o Relatório PEDE 2022 (p. 133) registra que, na autoavaliação, a nota
  zero significa a não participação do estudante, por escolha própria. O `iaa` original é preservado (é
  dele que o INDE oficial se reproduz), e a limpeza acrescenta `iaa_val`, em que a recusa vira ausência, e
  `iaa_recusa`, que guarda o comportamento como variável de análise — 249 casos, concentrados em 2023
  (4,5% / 20,0% / 1,9%). Toda média ou correlação de autoavaliação no EDA usa `iaa_val`.
- A `pedra` é normalizada para as quatro faixas oficiais (Quartzo, Ágata, Ametista, Topázio): a grafia de
  "Ágata" alterna entre acentuada e não acentuada de um ano para o outro, e o rótulo "INCLUIR" — que não é
  pedra, e sim marcação de pendência da planilha — vira NaN.
- `defasado` (defasagem < 0) já é marcado no painel — o flag registra o estado **atual** do aluno (não
  confundir com `atingiu_pv`, alvo distinto); o "risco" da pergunta 9 surge quando ele é deslocado para
  t+1 na modelagem. Os casos da fase anômala "9" de 2024 ficam False como os demais — o IAN bruto desses
  alunos é 10 (em fase), então não há razão para tratá-los como defasados.
- A evasão é calculada por diferença de conjuntos de `id_aluno` entre anos (presente no ano N, ausente no
  N+1), gerando `evasao.csv` — não se usa a coluna `Ativo/Inativo`, que é constante.

## 2. Análise (`notebooks/02_eda.ipynb`)

O notebook `02_eda.ipynb` parte do painel unificado (`pede_unificado.csv` e `evasao.csv`), deriva os
agregados de cada pergunta diretamente nas células e responde as 11 perguntas uma a uma.

As perguntas P1 a P7 são respondidas em **duas camadas**. A transversal compara alunos diferentes ano a
ano — é o que uma média anual mostra. A longitudinal acompanha o **mesmo aluno** de um ano para o outro,
no painel de 1.365 transições t→t+1 (897 alunos, 54% da base). A segunda é a que decide, porque a
evasão não é aleatória: quem sai tem indicadores piores, então a média anual pode subir sem que nenhum
aluno tenha melhorado. O aluno acompanhado de um ano para o seguinte, e não a coorte fixa dos três anos, é a unidade de
análise —
cobre 897 alunos contra 468 e não descarta quem entrou depois de 2022.

Os achados que ancoram o resto do projeto:

- A defasagem cai ano a ano no agregado (69,9% → 56,9% → 49,2%) e cai ainda mais forte na coorte fixa de
  alunos presentes nos três anos (67,3% → 34,8%). A queda **não** é efeito de composição: dentro do mesmo
  aluno e sobre o total de alunos acompanhados, 20,9% recuperam a fase ideal no ano seguinte contra 10,9% que a perdem —
  dois para um —, e o destino do conjunto melhora de uma transição para a outra: em 2022→2023, 61% dos
  alunos permaneceram defasados ou se tornaram defasados; em 2023→2024, apenas 42,6%. A taxa conta os
  universitários vindos das fases 0–7 (chegar à graduação é o desfecho perseguido), mas exclui quem entrou
  na associação já universitário — bolsistas, que nunca poderiam estar defasados.
- O IDA, ao contrário, **não está em alta**: a série anual sobe (6,09 → 6,66 → 6,35), mas 37,2% da alta de
  2022 para 2023 vem de quem saiu, e não de quem ficou — contando só quem voltou no ano seguinte, ela cai de
  +0,57 para +0,36. Dentro do mesmo aluno a variação é +0,12 em 2022→2023 (mediana 0,00, um empate) e −0,48
  em 2023→2024. Promover mais alunos explica o formato dessa queda, não o tamanho dela: congelando a
  proporção de promovidos, 87% da queda continua lá, e a mudança de mistura responde por 14%. O que a
  concentra é a fase de chegada — quem chegou à fase 3 em 2024 caiu −1,25, sobre 167 alunos.
- O engajamento (IEG) é o único achado que sobrevive intacto às duas camadas — anda junto com IDA e IPV
  tanto entre alunos (r ≈ 0,54) quanto dentro do mesmo aluno (Δ IEG × Δ IDA = 0,27 e 0,42). É a alavanca
  de intervenção mais defensável do notebook.
- A evasão fica em 30,2% (2022→2023) e 24,6% (2023→2024). A rigor a métrica mede "não retorno à avaliação
  PEDE seguinte", e o rótulo importa: parte dos ausentes pode ter saído por conclusão ou motivo externo,
  não por abandono.
- Os sinais univariados mais fortes contra o risco futuro são IPP, a própria defasagem atual, IPV e INDE —
  o que depois se confirma na importância de features do modelo.

Quatro leituras que a camada transversal sugeria **não passam** no teste longitudinal, e ficam registradas
no notebook: o IDA em alta (é seleção), o IPS como sinal antecedente de queda (é artefato de empilhar
anos — 28,0% do IPS de 2023 está no piso da escala), a queda de autoimagem em 2023 (é a recusa de
autoavaliação, não a nota) e o IPP como indicador que responde ao progresso de fase (é reversão à média).

## 3. Features e modelagem (`scripts/03_features.py`, `notebooks/04_modelagem.ipynb`)

A base do modelo (`base_risco.csv`, 1.290 transições) segue o desenho **t→t+1**: cada linha acompanha
um aluno de um ano para o seguinte, com as features observadas no ano t e o alvo `defasado` observado em t+1. Não
há circularidade — defasagem e indicadores do ano t são preditores legítimos do risco futuro; o que não pode é
usar os do próprio ano-alvo. Entram 16 features numéricas (indicadores, fase, idade, tempo de vínculo e as
tendências d_inde/d_ipv/d_ieg/d_ida, que são NaN quando não existe ano anterior) e 2 categóricas (gênero,
tipo de instituição). O IAN fica de fora por ser função determinística de defasagem — seria redundante.
Os quatro indicadores que ganham tendência são os de sinal mais forte na P9 do EDA (INDE −0,34, IPV −0,34,
IEG −0,22, IDA −0,21). O IAA fica em 0,02 — ruído, não justifica a coluna. O IPS marca +0,16 e entra só em
nível, o que a revisão do EDA acabou confirmando por outro caminho: a P5 mostra que a escala do IPS não é
comparável entre anos (28,0% dos alunos de 2023 no piso, contra 0,1% em 2022), de modo que uma diferença
ano a ano do próprio IPS mediria mudança de régua, não trajetória do aluno. E o IPP de 2022 é valor
reconstruído, cuja diferença para 2023 misturaria reconstruído com medido.

**Tratamento da autoavaliação.** O `iaa` publicado traz zero onde o aluno se recusou a responder: 249 linhas
em que o zero significa não participar, e não autoimagem baixa. Alimentar o modelo com esse zero é ensinar
uma queda de indicador que nunca aconteceu, então a autoavaliação entra em duas colunas — a nota como
`iaa_val`, em que a recusa vira ausente e é imputada pela mediana do treino, e o comportamento como
`iaa_recusa`, feature binária própria. Separar as duas se justifica porque a recusa carrega sinal
independente: em 2022 ela apareceu associada à evasão (51,3% contra 29,2%, p = 0,006). O efeito nas
métricas agregadas é modesto, como se espera do indicador de sinal univariado mais fraco da P9 (0,02), mas
o recall no subgrupo crítico sobe de 0,73 para 0,82 — e é esse subgrupo que justifica o modelo existir.

O split de teste separa 20% dos **alunos** (GroupShuffleSplit por `id_aluno`), não 20% das linhas, porque
o mesmo aluno em anos diferentes nos dois lados do split inflaria as métricas. Imputação e escala são
ajustadas só no treino. Três modelos disputam:

| Modelo | Acurácia | F1 | ROC-AUC | Recall novos defasados* |
|---|---|---|---|---|
| Regressão Logística (baseline) | 0,732 | 0,737 | 0,846 | 0,27 |
| **RandomForest (campeão)** | **0,789** | **0,806** | **0,873** | **0,82** |
| Rede Neural (Keras) | 0,762 | 0,777 | 0,837 | 0,68 |

\* recall no subgrupo crítico do teste: alunos em dia no ano t que se defasaram em t+1 (22 casos) —
exatamente quem um alerta precoce existe para pegar.

O RandomForest vence em todas as métricas agregadas e, sobretudo, no subgrupo crítico: pega 82% dos alunos
que vão se defasar estando hoje em dia, contra 27% do baseline linear. A rede neural (MLP com duas camadas
ocultas, dropout e early stopping por AUC em validação separada por aluno) fica próxima e cumpre o
requisito de Deep Learning da fase; com ~1.300 exemplos tabulares, o teto de uma MLP contra árvores é
esperado.

## 4. Validações — sem pressupor que os números estão certos

**Evasão.** A taxa de 20–30% parecia alta, então foi checada contra três hipóteses alternativas antes de
ser aceita: não é artefato de formatura (a fase 8, a última, tem a **menor** evasão da base, 6,3% sobre 63
alunos); nas demais fases o desenho é um degrau, e não um pico — as fases 0 a 2 ficam entre 23,3% e 25,4% e
da fase 3 em diante nenhuma fica abaixo de 29,5%, com a idade dizendo a mesma coisa (23,9% e 23,6% até os
12 anos, 30,4% dos 13 aos 15, 33,5% dos 16 em diante), e a fase 6, que marca 43,1%, são 22 alunos entre 51,
com as vizinhas em 33,6% e 29,5%; as saídas são permanentes (das 509 saídas marcadas, 4 reaparecem em algum
ano posterior, 0,8%); e os ids são inteiros sequenciais estáveis, sem sinal de recodificação entre anos. O
número se sustenta — com o caveat de rótulo já citado.

**Baseline de persistência.** A regra "quem está defasado hoje continua defasado no ano que vem" não exige
modelo algum e já acerta bastante (acurácia 0,686; ROC-AUC 0,744 usando −defasagem como score). O campeão
supera com folga (0,789 e 0,873) — o pipeline agrega sinal real além do óbvio.

**Validação temporal.** O split por aluno impede memorizar alunos, mas não impede aprender padrões
específicos de um ano; treinar só nas transições 2022→2023 e testar nas de 2023→2024 reproduz o cenário real
de uso. O poder de **ordenação** se mantém quase intacto (ROC-AUC 0,864 contra 0,873 do split aleatório),
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
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# a numeração é uma série única e reflete a ordem de execução: a limpeza alimenta o EDA, o EDA
# fundamenta a escolha de features e o script de features alimenta a modelagem
python scripts/01_clean.py
jupyter notebook notebooks/02_eda.ipynb
python scripts/03_features.py
jupyter notebook notebooks/04_modelagem.ipynb  # exporta os artefatos de models/

# dashboard
streamlit run dashboard/app.py
```

Os artefatos de `models/` já estão versionados, então o dashboard roda sem re-treinar nada; re-executar
`04_modelagem.ipynb` regenera os mesmos arquivos (seeds fixas, SEED=42).

São dois arquivos de dependências, de propósito. O `requirements-dev.txt` acima é o ambiente completo,
único que roda a pipeline e os notebooks — soma openpyxl (leitura do `.xlsx` bruto), jupyter, tensorflow
(a rede neural do `04_modelagem`) e kaleido (exportação estática dos gráficos). Já o `requirements.txt`
tem só o que o dashboard importa: é o arquivo que o Streamlit Community Cloud lê na raiz no momento do
deploy, e enxugá-lo evita instalar centenas de MB que a aplicação nunca usa. As versões dele estão
pinadas porque o `risk_rf.pkl` foi serializado com scikit-learn 1.9.0 — pickle de sklearn não tem
compatibilidade garantida entre versões. O ambiente local é Python 3.11; selecionar a mesma versão nas
configurações avançadas do deploy.
