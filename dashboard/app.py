"""
Dashboard do Datathon Passos Mágicos (FIAP Pós-Tech, Fase 5). Duas seções: a visão geral do painel PEDE
2022-2024, que carrega os achados centrais do EDA, e o preditor de risco de defasagem, que é o objetivo do
app — servido pelo modelo campeão (RandomForest, ROC-AUC 0.873 no teste), treinado no desenho early-warning
t→t+1: os indicadores do ano corrente preveem se o aluno estará defasado (defasagem < 0) no ano seguinte.

A leitura por fase e por indicador que existia numa terceira seção saiu: ela repetia, em versão mais pobre,
o que o `notebooks/02_eda.ipynb` e a apresentação de storytelling já entregam com mais espaço e contexto.

O preditor cobre apenas as fases 0-7: a fase 8 (universitários) é avaliada por rubrica própria, sem os
indicadores que alimentam o modelo, e por isso fica fora tanto do treino quanto da predição. Pelo mesmo
motivo, o IAN e o INDE não são pedidos ao usuário — o IAN é função determinística da defasagem e o INDE é
recalculado pela fórmula oficial das fases 0-7, o que impede a entrada de combinações inconsistentes.

Execução local (a partir da raiz do repositório): streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
PAINEL_PATH = RAIZ / "data" / "processed" / "pede_unificado.csv"
EVASAO_PATH = RAIZ / "data" / "processed" / "evasao.csv"
MODELO_PATH = RAIZ / "models" / "risk_rf.pkl"
METADATA_PATH = RAIZ / "models" / "risk_metadata.json"

PESOS_INDE_0_7 = {"ian": 0.1, "ida": 0.2, "ieg": 0.2, "iaa": 0.1, "ips": 0.1, "ipp": 0.1, "ipv": 0.2}

# Paleta da marca pessoal, em ordem fixa validada para daltonismo e visão normal sobre o fundo #F5F7FA:
# âmbar e coral não podem ficar adjacentes (ΔE 11.9, abaixo do piso de distinção de 15).
CORES_MARCA = ["#1F3A5F", "#E0A458", "#2F6F73", "#DF745C", "#4C6A92", "#9AA5B1"]
COR_TINTA = "#1F3A5F"
COR_GRADE = "rgba(154, 165, 177, 0.35)"  # Cool Gray da marca com transparência

px.defaults.color_discrete_sequence = CORES_MARCA

st.set_page_config(page_title="Datathon Passos Mágicos", page_icon="🎓", layout="wide")


def aplicar_marca(fig):
    """Aplica a identidade visual sobre o gráfico Plotly: fundos transparentes (a superfície off-white vem
    do tema do Streamlit em .streamlit/config.toml), tipografia na tinta da marca e grade discreta em Cool
    Gray — a grade deve recuar, não competir com as marcas de dados."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COR_TINTA,
        title_font_color=COR_TINTA,
        legend_title_font_color=COR_TINTA,
    )
    fig.update_xaxes(gridcolor=COR_GRADE, linecolor=COR_GRADE, zerolinecolor=COR_GRADE)
    fig.update_yaxes(gridcolor=COR_GRADE, linecolor=COR_GRADE, zerolinecolor=COR_GRADE)
    return fig


@st.cache_data
def carregar_painel() -> pd.DataFrame:
    df = pd.read_csv(PAINEL_PATH)
    df["defasado"] = (
        df["defasado"].map(lambda v: pd.NA if pd.isna(v) else v in (True, "True")).astype("boolean")
    )
    return df


@st.cache_data
def carregar_evasao() -> pd.DataFrame:
    return pd.read_csv(EVASAO_PATH)


@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(MODELO_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return modelo, meta


def ian_a_partir_de_defas(defasagem: int) -> float:
    """O IAN é função determinística da defasagem: 2.5 quando severa (defasagem <= -3), 5 quando moderada
    (defasagem -2 ou -1) e 10 quando o aluno está em fase (defasagem >= 0). Derivá-lo da defasagem informada evita
    que o usuário entre com um par (defasagem, IAN) que não existe nos dados."""
    if defasagem <= -3:
        return 2.5
    if defasagem < 0:
        return 5.0
    return 10.0


def base_defasagem(df: pd.DataFrame) -> pd.DataFrame:
    """Recorte usado em toda taxa de defasagem, idêntico ao da P1 do 02_eda.ipynb. Universitários (fase 8 e
    a fase anômala "9", que fica com fase NaN) só entram se já passaram pelas fases 0-7 em algum ano
    anterior: chegar à graduação vindo do programa é o desfecho que a taxa deve capturar, mas quem entrou na
    associação já universitário é bolsista — nunca poderia estar defasado e só diluiria a série. Precisa do
    painel inteiro, não de um corte por ano, porque a primeira passagem pelas fases 0-7 é histórica."""
    uni = df["fase"].isna() | (df["fase"] == 8)
    prim_07 = df.loc[~uni].groupby("id_aluno")["ano"].min()
    return df[~uni | (df["id_aluno"].map(prim_07) < df["ano"])]


ORDEM_TRAJETORIA = ["Segue em fase", "Recupera a fase ideal", "Perde a fase ideal", "Segue defasado"]
CORES_TRAJETORIA = {
    "Segue em fase": "#1F3A5F", "Recupera a fase ideal": "#2F6F73",
    "Perde a fase ideal": "#DF745C", "Segue defasado": "#9AA5B1",
}
ORDEM_SAIDA = ["Permaneceu", "Saiu no ano seguinte"]
CORES_SAIDA = {"Permaneceu": "#1F3A5F", "Saiu no ano seguinte": "#DF745C"}


def taxa_defasagem_por_ano(df: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """Percentual de defasados por ano, em formato longo e com o rótulo do grupo — o `defasado` é booleano
    anulável do pandas, e o Plotly precisa de float puro."""
    serie = df.groupby("ano")["defasado"].mean().mul(100).astype(float)
    return serie.rename("pct").reset_index().assign(grupo=grupo)


def trajetoria_por_transicao(df: pd.DataFrame, anos: list) -> tuple[pd.DataFrame, int]:
    """Reparte cada transição ano→ano+1 nos quatro destinos possíveis do mesmo aluno. Só entram alunos com
    avaliação nas duas pontas: é o recorte que torna a queda de defasagem um movimento de pessoa, e não um
    efeito de quem entrou ou saiu da base entre um ano e outro."""
    largura = df.pivot_table(index="id_aluno", columns="ano", values="defasado", aggfunc="first")
    linhas, total = [], 0
    for ano in anos[:-1]:
        if ano + 1 not in largura.columns:
            continue
        par = largura[[ano, ano + 1]].dropna()
        antes, depois = par[ano].astype(bool), par[ano + 1].astype(bool)
        total += len(par)
        destinos = {
            "Segue em fase": ~antes & ~depois, "Recupera a fase ideal": antes & ~depois,
            "Perde a fase ideal": ~antes & depois, "Segue defasado": antes & depois,
        }
        linhas += [{"transicao": f"{ano}→{ano + 1}", "situacao": nome, "pct": mascara.mean() * 100}
                   for nome, mascara in destinos.items()]
    return pd.DataFrame(linhas), total


def perfil_de_saida(df: pd.DataFrame, evasao: pd.DataFrame) -> pd.DataFrame:
    """Média de IEG, IDA e IPS no ano de referência, separando quem reapareceu na avaliação seguinte de quem
    não reapareceu. O ano vira texto porque o Plotly trataria a coluna numérica como escala contínua."""
    perfil = df.merge(evasao, left_on=["id_aluno", "ano"], right_on=["id_aluno", "ano_referencia"])
    perfil["grupo"] = perfil["evadiu"].map({True: "Saiu no ano seguinte", False: "Permaneceu"})
    medias = perfil.groupby(["ano_referencia", "grupo"])[["ieg", "ida", "ips"]].mean().reset_index()
    longo = medias.melt(id_vars=["ano_referencia", "grupo"], var_name="indicador", value_name="media")
    return longo.assign(ano=lambda d: d["ano_referencia"].astype(str),
                        indicador=lambda d: d["indicador"].str.upper())


def pagina_visao_geral(df: pd.DataFrame, evasao: pd.DataFrame) -> None:
    st.header("Visão geral — PEDE 2022-2024")
    st.markdown(
        "Os três achados que sustentam o resto da entrega: a defasagem cai, ela cai **dentro do mesmo "
        "aluno**, e a base se renova o bastante para que nenhuma média anual possa ser lida sozinha."
    )

    anos = sorted(df["ano"].unique())
    ano_sel = st.selectbox("Ano de referência dos cartões", anos, index=len(anos) - 1)
    corte = df[df["ano"] == ano_sel]

    df_def = base_defasagem(df)
    corte_def = df_def[df_def["ano"] == ano_sel]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alunos avaliados", f"{corte['id_aluno'].nunique()}")
    c2.metric(
        "Alunos defasados", f"{corte_def['defasado'].mean() * 100:.1f}%",
        help="Sobre os alunos que passaram pelas fases 0-7 — bolsistas que entraram já universitários "
             "ficam fora, como na P1 do EDA.",
    )
    c3.metric(
        "Defasagem severa", f"{(corte_def['defasagem'] <= -3).mean() * 100:.1f}%",
        help="Três anos ou mais atrás da fase ideal. É o movimento mais forte da P1: 3,3% em 2022, 1,4% em "
             "2023 e 0,3% em 2024 — o atraso extremo praticamente desaparece do painel.",
    )
    # O último ano do painel não tem ano seguinte para comparar: o cartão fica vazio de propósito, e o
    # aviso vai no corpo dele para que a lacuna se explique sem depender do tooltip.
    ev_ano = evasao[evasao["ano_referencia"] == ano_sel]["evadiu"]
    c4.metric(
        f"Evasão {ano_sel}→{ano_sel + 1}",
        f"{ev_ano.mean() * 100:.1f}%" if len(ev_ano) else "—",
        delta=None if len(ev_ano) else "sem avaliação posterior na base",
        delta_color="off",
        help="Aluno presente no ano de referência e ausente no seguinte. Não há transição a partir de 2024.",
    )

    col_esq, col_dir = st.columns(2)

    # A coorte fixa é o teste mais duro da queda: sempre as mesmas pessoas, sem entrada nem saída para
    # explicar o resultado. Definida dentro do recorte da P1, não sobre o painel bruto.
    anos_por_aluno = df_def.groupby("id_aluno")["ano"].nunique()
    coorte = anos_por_aluno[anos_por_aluno == len(anos)].index
    serie = pd.concat([
        taxa_defasagem_por_ano(df_def, "Painel do ano (todos os avaliados)"),
        taxa_defasagem_por_ano(df_def[df_def["id_aluno"].isin(coorte)],
                               f"Coorte fixa ({len(coorte)} alunos, presentes nos três anos)"),
    ])
    fig = px.line(
        serie, x="ano", y="pct", color="grupo", markers=True,
        title="Alunos defasados por ano (%): painel × coorte fixa",
        labels={"ano": "Ano", "pct": "% defasados", "grupo": ""},
    )
    fig.update_traces(line_width=2, marker_size=9)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 80])
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    col_esq.plotly_chart(aplicar_marca(fig), width="stretch")
    col_esq.caption(
        "A defasagem cai de 69,9% para 49,2% no painel. Na coorte fixa — sempre os mesmos 468 alunos — ela "
        "cai mais, de 67,3% para 34,8%: o resultado é maior justamente no grupo em que a composição não "
        "muda. A taxa publicada é a mais alta das duas porque inclui quem chegou depois — dos 358 alunos "
        "que entraram em 2024, 62,8% já chegaram defasados."
    )

    traj, n_transicoes = trajetoria_por_transicao(df_def, anos)
    fig = px.bar(
        traj, x="transicao", y="pct", color="situacao", text_auto=".1f",
        category_orders={"situacao": ORDEM_TRAJETORIA}, color_discrete_map=CORES_TRAJETORIA,
        title="Para onde vai o mesmo aluno, de um ano para o outro (%)",
        labels={"transicao": "Transição", "pct": "% dos alunos acompanhados", "situacao": ""},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    col_dir.plotly_chart(aplicar_marca(fig), width="stretch")
    total_fmt = f"{n_transicoes:,}".replace(",", ".")  # separador de milhar em português
    col_dir.caption(
        f"Aqui a queda deixa de ser estatística de turma e vira movimento de pessoa. Nas {total_fmt} "
        "transições acompanhadas, 20,9% dos alunos recuperam a fase ideal contra 10,9% que a perdem — dois "
        "recuperam para cada um que perde. E o ritmo melhora: 17,5% de recuperação em 2022→2023 contra "
        "23,7% em 2023→2024."
    )

    fig = px.bar(
        perfil_de_saida(df, evasao), x="indicador", y="media", color="grupo", barmode="group",
        text_auto=".2f", facet_col="ano", category_orders={"grupo": ORDEM_SAIDA},
        color_discrete_map=CORES_SAIDA,
        title="Quem sai já estava atrás — média do indicador no ano de referência",
        labels={"indicador": "", "media": "Média (0-10)", "grupo": ""},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(range=[0, 10])
    st.plotly_chart(aplicar_marca(fig), width="stretch")

    st.caption(
        "Entre um quarto e um terço da base se renova a cada ano — 30,2% dos alunos de 2022 não estavam lá "
        "em 2023, e 24,6% dos de 2023 não estavam em 2024. E quem sai não é um aluno qualquer: sai com "
        "engajamento e desempenho visivelmente abaixo de quem fica. É por isso que uma média anual pode "
        "subir sem que nenhum aluno tenha melhorado, e é por isso que os dois gráficos acima comparam o "
        "aluno com ele mesmo: a alta do IDA em 2023 (6,09 → 6,66), por exemplo, encolhe de +0,57 para "
        "+0,36 contando só quem voltou no ano seguinte, e dentro do mesmo aluno vira empate (+0,12) e "
        "depois queda (−0,48). Repare também no IPS: é o único dos três que não separa quem sai de quem "
        "fica — o mesmo indicador que a P5 mostra não servir como sinal antecedente."
    )
    st.caption(
        "⚠️ *Sair* aqui significa não reaparecer na avaliação PEDE seguinte, o que não é necessariamente "
        "abandono do programa. A análise completa — por fase, por indicador e pergunta a pergunta — está "
        "em `notebooks/02_eda.ipynb`."
    )


def pagina_preditor(modelo, meta: dict) -> None:
    st.header("Preditor de risco de defasagem")
    st.markdown(
        "Informe os dados do aluno **no ano corrente**; o modelo estima a probabilidade de ele estar "
        "**defasado no ano seguinte** (desenho early-warning t→t+1). Válido para as fases 0-7 — a fase 8 "
        "é avaliada por rubrica própria e não é coberta pelo modelo."
    )

    with st.form("form_preditor"):
        c1, c2, c3 = st.columns(3)
        fase = c1.selectbox("Fase atual", list(range(8)))
        defasagem = c2.slider(
            "Defasagem atual (fase − fase ideal)", -5, 2, 0,
            help="Negativo = aluno atrasado em relação à fase ideal para a idade.",
        )
        idade = c3.slider("Idade", 6, 25, 12)

        c1, c2, c3 = st.columns(3)
        tempo_vinculo = c1.slider("Anos de vínculo com a associação", 0, 10, 1)
        genero = c2.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)
        tipo_inst = c3.selectbox("Tipo de instituição de ensino", ["Pública", "Privada"])

        st.markdown("**Indicadores do ano corrente (0-10)** — o IAN e o INDE são derivados automaticamente.")
        c1, c2, c3 = st.columns(3)
        ida = c1.slider("IDA — desempenho acadêmico", 0.0, 10.0, 7.0, 0.1)
        ieg = c2.slider("IEG — engajamento", 0.0, 10.0, 8.0, 0.1)
        ipv = c3.slider("IPV — ponto de virada", 0.0, 10.0, 7.5, 0.1)
        c1, c2, c3 = st.columns(3)
        iaa = c1.slider("IAA — autoavaliação", 0.0, 10.0, 8.5, 0.1)
        ips = c2.slider("IPS — psicossocial", 0.0, 10.0, 7.0, 0.1)
        ipp = c3.slider("IPP — psicopedagógico", 0.0, 10.0, 7.5, 0.1)

        # Na base PEDE o zero do IAA é recusa de responder, não autoimagem baixa (relatório 2022, p. 133),
        # e as duas coisas entram no modelo separadas: a nota como iaa_val (ausente na recusa, imputada pelo
        # pipeline) e o comportamento como iaa_recusa. O INDE, esse sim, usa o zero — é assim que a
        # associação calcula o índice oficial, com a penalidade embutida.
        iaa_recusa = st.checkbox(
            "O aluno se recusou a responder à autoavaliação",
            help="Marque em vez de arrastar o IAA para zero. A recusa é comportamento, não nota: o modelo "
                 "a recebe como sinal próprio, e o INDE é recalculado com a penalidade oficial (IAA = 0).",
        )

        tem_ano_anterior = st.checkbox(
            "O aluno tem avaliação do ano anterior (habilita as tendências)", value=False,
            help="Sem o ano anterior as variações ficam ausentes e o modelo usa a mediana do treino.",
        )
        c1, c2, c3, c4 = st.columns(4)
        d_inde = c1.number_input("Δ INDE (ano atual − anterior)", -10.0, 10.0, 0.0, 0.1)
        d_ipv = c2.number_input("Δ IPV", -10.0, 10.0, 0.0, 0.1)
        d_ieg = c3.number_input("Δ IEG", -10.0, 10.0, 0.0, 0.1)
        d_ida = c4.number_input("Δ IDA", -10.0, 10.0, 0.0, 0.1)

        limiar = st.slider(
            "Limiar de classificação", 0.05, 0.95, float(meta["limiar_padrao"]), 0.05,
            help="Limiar menor prioriza recall (menos alunos em risco passam despercebidos); "
                 "maior prioriza precisão (menos alertas falsos).",
        )

        submetido = st.form_submit_button("Estimar risco", type="primary")

    if not submetido:
        return

    ian = ian_a_partir_de_defas(defasagem)
    indicadores = {"ian": ian, "ida": ida, "ieg": ieg, "iaa": 0.0 if iaa_recusa else iaa,
                   "ips": ips, "ipp": ipp, "ipv": ipv}
    inde = sum(PESOS_INDE_0_7[k] * v for k, v in indicadores.items())

    entrada = pd.DataFrame([{
        "defasagem": float(defasagem), "inde": inde, "ida": ida, "ieg": ieg,
        "iaa_val": np.nan if iaa_recusa else iaa, "iaa_recusa": float(iaa_recusa), "ips": ips,
        "ipp": ipp, "ipv": ipv, "fase": float(fase), "idade": float(idade),
        "tempo_vinculo": float(tempo_vinculo),
        "d_inde": d_inde if tem_ano_anterior else np.nan,
        "d_ipv": d_ipv if tem_ano_anterior else np.nan,
        "d_ieg": d_ieg if tem_ano_anterior else np.nan,
        "d_ida": d_ida if tem_ano_anterior else np.nan,
        "genero": "Masc" if genero == "Masculino" else "Fem",
        "tipo_instituicao": tipo_inst,
    }])

    proba = float(modelo.predict_proba(entrada)[0, 1])
    em_risco = proba >= limiar

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidade de defasagem em t+1", f"{proba * 100:.1f}%")
    c2.metric("IAN derivado / INDE calculado", f"{ian:.1f} / {inde:.2f}")
    c3.metric("Classificação", "🔴 Em risco" if em_risco else "🟢 Fora de risco",
              help=f"Limiar aplicado: {limiar:.2f}")
    st.progress(min(proba, 1.0))

    if iaa_recusa:
        sem_iaa = {k: p for k, p in PESOS_INDE_0_7.items() if k != "iaa"}
        inde_sem_penalidade = sum(sem_iaa[k] * indicadores[k] for k in sem_iaa) / sum(sem_iaa.values())
        st.caption(
            f"Com a recusa, o INDE oficial cai para {inde:.2f}; repondo os seis indicadores restantes para "
            f"somar 1, ele seria {inde_sem_penalidade:.2f}. Na base, essa penalidade fez 148 dos 245 alunos "
            "que recusaram — 60,4% — aparecerem uma pedra abaixo (P4 do EDA)."
        )

    if em_risco:
        st.error(
            "Aluno com risco relevante de defasagem no próximo ano — vale priorizar acompanhamento "
            "psicopedagógico e ações de engajamento antes da próxima avaliação."
        )
    else:
        st.success(
            "Risco baixo no cenário informado. Ainda assim, vale acompanhar a evolução do IEG e do IDA ao "
            "longo do ano: quedas nesses indicadores antecedem parte das novas defasagens."
        )

    with st.expander("Desempenho dos modelos no conjunto de teste (261 avaliações de alunos nunca vistos)"):
        tab = pd.DataFrame(meta["metricas_teste"]).T.round(3)
        st.dataframe(tab, width="stretch")
        st.caption(
            f"Campeão em produção: **{meta['campeao']}**. A rede neural (Keras) e a regressão logística "
            "estão treinadas e comparadas em notebooks/04_modelagem.ipynb."
        )


def main() -> None:
    df = carregar_painel()
    evasao = carregar_evasao()
    modelo, meta = carregar_modelo()

    st.sidebar.title("🎓 Passos Mágicos")
    pagina = st.sidebar.radio(
        "Seção",
        ["Visão geral", "Preditor de risco"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Datathon FIAP Pós-Tech — Fase 5**\n\n"
        "Painel PEDE 2022-2024: 1.661 alunos, 3.030 avaliações.\n\n"
        "Modelo early-warning: prevê a defasagem do ano seguinte a partir dos indicadores do ano corrente "
        f"({meta['campeao']}, ROC-AUC {meta['metricas_teste'][meta['campeao']]['roc_auc']:.3f} no teste)."
    )

    if pagina == "Visão geral":
        pagina_visao_geral(df, evasao)
    else:
        pagina_preditor(modelo, meta)


if __name__ == "__main__":
    main()
