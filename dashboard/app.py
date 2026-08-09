"""
Dashboard do Datathon Passos Mágicos (FIAP Pós-Tech, Fase 5). Três seções: visão geral do painel PEDE
2022-2024, análise por fase e indicadores, e o preditor de risco de defasagem — este último servido pelo
modelo campeão (RandomForest, ROC-AUC 0.871 no teste), treinado no desenho early-warning t→t+1: os
indicadores do ano corrente preveem se o aluno estará defasado (defasagem < 0) no ano seguinte.

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
ESCALA_RISCO = ["#F5F7FA", "#E79887", "#DF745C"]  # sequencial de um matiz só: claro = pouco risco
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


def pagina_visao_geral(df: pd.DataFrame, evasao: pd.DataFrame) -> None:
    st.header("Visão geral — PEDE 2022-2024")

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
    c3.metric("INDE médio", f"{corte['inde'].mean():.2f}")
    ev_ano = evasao[evasao["ano_referencia"] == ano_sel]["evadiu"]
    c4.metric(
        f"Evasão {ano_sel}→{ano_sel + 1}",
        f"{ev_ano.mean() * 100:.1f}%" if len(ev_ano) else "—",
        help="Aluno presente no ano de referência e ausente no seguinte. Não há transição a partir de 2024.",
    )

    col_esq, col_dir = st.columns(2)

    serie_risco = df_def.groupby("ano")["defasado"].mean().mul(100).reset_index(name="pct")
    fig = px.line(
        serie_risco, x="ano", y="pct", markers=True,
        title="Alunos defasados por ano (%)",
        labels={"ano": "Ano", "pct": "% defasados"},
    )
    fig.update_traces(line_width=2, marker_size=9)
    fig.update_xaxes(dtick=1)
    col_esq.plotly_chart(aplicar_marca(fig), width="stretch")

    medias = df.groupby("ano")[["inde", "ida", "ieg", "ipv"]].mean().round(2).reset_index()
    medias_long = medias.melt(id_vars="ano", var_name="indicador", value_name="media")
    medias_long["indicador"] = medias_long["indicador"].str.upper()
    fig = px.bar(
        medias_long, x="ano", y="media", color="indicador", barmode="group",
        title="Médias dos principais indicadores por ano",
        labels={"ano": "Ano", "media": "Média (0-10)", "indicador": "Indicador"},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(dtick=1)
    col_dir.plotly_chart(aplicar_marca(fig), width="stretch")

    st.caption(
        "A queda consistente da defasagem (69,9% → 56,9% → 49,2%) é o achado central do painel: na coorte "
        "fixa dos alunos presentes nos três anos, a taxa cai de 67,3% para 34,8%, e acompanhando aluno a "
        "aluno 34,2% dos defasados recuperam a fase ideal no ano seguinte contra 27,9% que a perdem — a "
        "queda é real, não efeito de quem saiu da base. A série conta os alunos que passaram pelas fases "
        "0-7 e deixa de fora os bolsistas que entraram na associação já universitários — nunca poderiam "
        "estar defasados. É o mesmo recorte da P1 do EDA."
    )
    st.caption(
        "⚠️ As médias anuais dos indicadores comparam alunos diferentes a cada ano e devem ser lidas com "
        "cuidado: a alta do IDA em 2023, por exemplo, encolhe de +0,57 para +0,36 quando se conta só quem "
        "voltou no ano seguinte — 37,2% dela vem de quem saiu, não de quem ficou. Dentro do mesmo aluno, "
        "2022→2023 é empate (+0,12) e 2023→2024 cai (−0,48). Ver P2 do EDA."
    )


def pagina_analise(df: pd.DataFrame) -> None:
    st.header("Análise por fase e indicadores")

    # ano vira categórico para os gráficos coloridos por ano: numérico, o Plotly aplicaria uma escala
    # contínua em vez das cores discretas da marca.
    df07 = df[df["fase"] <= 7].assign(ano=lambda d: d["ano"].astype(str))

    col_esq, col_dir = st.columns(2)

    piv = (
        df07.pivot_table(index="fase", columns="ano", values="defasado", aggfunc="mean")
        .mul(100).round(1)
    )
    fig = px.imshow(
        piv, text_auto=".0f", aspect="auto", color_continuous_scale=ESCALA_RISCO,
        title="Alunos defasados por fase × ano (%)",
        labels={"x": "Ano", "y": "Fase", "color": "% defasados"},
    )
    # Mesmo com o ano como string, o Plotly reconverte rótulos numéricos para eixo linear e cria ticks
    # fracionários (2.022,5); forçar o tipo categórico mantém só os três anos.
    fig.update_xaxes(type="category")
    fig.update_yaxes(dtick=1)
    col_esq.plotly_chart(aplicar_marca(fig), width="stretch")

    fig = px.box(
        df07.dropna(subset=["inde"]), x="fase", y="inde", color="ano",
        title="Distribuição do INDE por fase",
        labels={"fase": "Fase", "inde": "INDE", "ano": "Ano"},
    )
    col_dir.plotly_chart(aplicar_marca(fig), width="stretch")

    amostra = df07.dropna(subset=["ieg", "ida"])
    fig = px.scatter(
        amostra, x="ieg", y="ida", color="ano", opacity=0.45,
        title="Engajamento (IEG) × desempenho acadêmico (IDA)",
        labels={"ieg": "IEG", "ida": "IDA", "ano": "Ano"},
    )
    st.plotly_chart(aplicar_marca(fig), width="stretch")

    st.caption(
        "A relação entre engajamento e desempenho é a mais sólida da análise: além da correlação entre "
        "alunos visível no gráfico (r ≈ 0,54), ela se confirma dentro do mesmo aluno — quando o IEG sobe "
        "de um ano para o outro, o IDA sobe junto (P3 do EDA). É a associação mais bem sustentada do "
        "relatório, o que não estabelece que aumentar o IEG *cause* a melhora do IDA — as duas coisas "
        "andam juntas, e a P3 não separa qual puxa qual. A fase 8 (universitários) fica fora destes "
        "gráficos: é avaliada por rubrica própria e não possui os indicadores padrão no painel — apenas "
        "o IAN e a defasagem."
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
        ["Visão geral", "Análise por fase", "Preditor de risco"],
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
    elif pagina == "Análise por fase":
        pagina_analise(df)
    else:
        pagina_preditor(modelo, meta)


if __name__ == "__main__":
    main()
