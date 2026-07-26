"""
Dashboard do Datathon Passos Mágicos (FIAP Pós-Tech, Fase 5). Três seções: visão geral do painel PEDE
2022-2024, análise por fase e indicadores, e o preditor de risco de defasagem — este último servido pelo
modelo campeão (RandomForest, ROC-AUC 0.871 no teste), treinado no desenho early-warning t→t+1: os
indicadores do ano corrente preveem se o aluno estará defasado (defas < 0) no ano seguinte.

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
    df["em_risco"] = (
        df["em_risco"].map(lambda v: pd.NA if pd.isna(v) else v in (True, "True")).astype("boolean")
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


def ian_a_partir_de_defas(defas: int) -> float:
    """O IAN é função determinística da defasagem: 2.5 quando severa (defas <= -3), 5 quando moderada
    (defas -2 ou -1) e 10 quando o aluno está em fase (defas >= 0). Derivá-lo da defasagem informada evita
    que o usuário entre com um par (defas, IAN) que não existe nos dados."""
    if defas <= -3:
        return 2.5
    if defas < 0:
        return 5.0
    return 10.0


def pagina_visao_geral(df: pd.DataFrame, evasao: pd.DataFrame) -> None:
    st.header("Visão geral — PEDE 2022-2024")

    anos = sorted(df["ano"].unique())
    ano_sel = st.selectbox("Ano de referência dos cartões", anos, index=len(anos) - 1)
    corte = df[df["ano"] == ano_sel]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alunos avaliados", f"{corte['id_aluno'].nunique()}")
    c2.metric("Alunos defasados", f"{corte['em_risco'].mean() * 100:.1f}%")
    c3.metric("INDE médio", f"{corte['inde'].mean():.2f}")
    ev_ano = evasao[evasao["ano_referencia"] == ano_sel]["evadiu"]
    c4.metric(
        f"Evasão {ano_sel}→{ano_sel + 1}",
        f"{ev_ano.mean() * 100:.1f}%" if len(ev_ano) else "—",
        help="Aluno presente no ano de referência e ausente no seguinte. Não há transição a partir de 2024.",
    )

    col_esq, col_dir = st.columns(2)

    serie_risco = df.groupby("ano")["em_risco"].mean().mul(100).reset_index(name="pct")
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
        "A queda consistente da defasagem (69,9% → 54,4% → 47,8%) é o achado central do painel: na coorte "
        "fixa dos alunos presentes nos três anos, a taxa cai de 67,3% para 34,8%."
    )


def pagina_analise(df: pd.DataFrame) -> None:
    st.header("Análise por fase e indicadores")

    # ano vira categórico para os gráficos coloridos por ano: numérico, o Plotly aplicaria uma escala
    # contínua em vez das cores discretas da marca.
    df07 = df[df["fase"] <= 7].assign(ano=lambda d: d["ano"].astype(str))

    col_esq, col_dir = st.columns(2)

    piv = (
        df07.pivot_table(index="fase", columns="ano", values="em_risco", aggfunc="mean")
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
        "A fase 8 (universitários) fica fora destes gráficos: é avaliada por rubrica própria e não possui "
        "os indicadores padrão no painel — apenas o IAN e a defasagem."
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
        defas = c2.slider(
            "Defasagem atual (fase − fase ideal)", -5, 2, 0,
            help="Negativo = aluno atrasado em relação à fase ideal para a idade.",
        )
        idade = c3.slider("Idade", 6, 25, 12)

        c1, c2, c3 = st.columns(3)
        tempo_vinculo = c1.slider("Anos de vínculo com a associação", 0, 10, 1)
        genero = c2.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)
        tipo_inst = c3.selectbox("Tipo de instituição de ensino", ["Pública", "Privada", "Outra"])

        st.markdown("**Indicadores do ano corrente (0-10)** — o IAN e o INDE são derivados automaticamente.")
        c1, c2, c3 = st.columns(3)
        ida = c1.slider("IDA — desempenho acadêmico", 0.0, 10.0, 7.0, 0.1)
        ieg = c2.slider("IEG — engajamento", 0.0, 10.0, 8.0, 0.1)
        ipv = c3.slider("IPV — ponto de virada", 0.0, 10.0, 7.5, 0.1)
        c1, c2, c3 = st.columns(3)
        iaa = c1.slider("IAA — autoavaliação", 0.0, 10.0, 8.5, 0.1)
        ips = c2.slider("IPS — psicossocial", 0.0, 10.0, 7.0, 0.1)
        ipp = c3.slider("IPP — psicopedagógico", 0.0, 10.0, 7.5, 0.1)

        tem_ano_anterior = st.checkbox(
            "O aluno tem avaliação do ano anterior (habilita as tendências)", value=False,
            help="Sem o ano anterior as variações ficam ausentes e o modelo usa a mediana do treino.",
        )
        c1, c2, c3 = st.columns(3)
        d_inde = c1.number_input("Δ INDE (ano atual − anterior)", -10.0, 10.0, 0.0, 0.1)
        d_ieg = c2.number_input("Δ IEG", -10.0, 10.0, 0.0, 0.1)
        d_ida = c3.number_input("Δ IDA", -10.0, 10.0, 0.0, 0.1)

        limiar = st.slider(
            "Limiar de classificação", 0.05, 0.95, float(meta["limiar_padrao"]), 0.05,
            help="Limiar menor prioriza recall (menos alunos em risco passam despercebidos); "
                 "maior prioriza precisão (menos alertas falsos).",
        )

        submetido = st.form_submit_button("Estimar risco", type="primary")

    if not submetido:
        return

    ian = ian_a_partir_de_defas(defas)
    indicadores = {"ian": ian, "ida": ida, "ieg": ieg, "iaa": iaa, "ips": ips, "ipp": ipp, "ipv": ipv}
    inde = sum(PESOS_INDE_0_7[k] * v for k, v in indicadores.items())

    entrada = pd.DataFrame([{
        "defas": float(defas), "inde": inde, "ida": ida, "ieg": ieg, "iaa": iaa, "ips": ips,
        "ipp": ipp, "ipv": ipv, "fase": float(fase), "idade": float(idade),
        "tempo_vinculo": float(tempo_vinculo),
        "d_inde": d_inde if tem_ano_anterior else np.nan,
        "d_ieg": d_ieg if tem_ano_anterior else np.nan,
        "d_ida": d_ida if tem_ano_anterior else np.nan,
        "genero": 1.0 if genero == "Masculino" else 0.0,
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
            "estão treinadas e comparadas em notebooks/02_modelagem.ipynb."
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
