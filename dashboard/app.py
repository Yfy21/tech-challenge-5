import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Passos Mágicos — Painel de Risco",
    page_icon="✨",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pede_unificado.csv")


@st.cache_data(ttl=300)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "risk_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Passos Mágicos ✨")
pagina = st.sidebar.radio("Navegação", ["Visão Geral", "Análise por Fase", "Previsão de Risco"])

df = load_data()
model = load_model()

# ── Visão Geral ───────────────────────────────────────────────────────────────
if pagina == "Visão Geral":
    st.title("Visão Geral — Indicadores 2022–2024")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de alunos", df["id_aluno"].nunique())
    col2.metric("Avaliações", len(df))
    col3.metric("Em risco", int(df["em_risco"].sum()))
    col4.metric("INDE médio", f"{df['inde'].mean():.2f}")

    st.subheader("Evolução do INDE por fase e ano")
    fig = px.line(
        df.groupby(["ano", "fase"])["inde"].mean().reset_index(),
        x="ano", y="inde", color="fase", markers=True,
        labels={"inde": "INDE médio", "ano": "Ano", "fase": "Fase"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribuição de alunos em risco por ano")
    risco_ano = df.groupby("ano")["em_risco"].mean().reset_index()
    risco_ano["em_risco"] *= 100
    fig2 = px.bar(risco_ano, x="ano", y="em_risco",
                  labels={"em_risco": "% em risco", "ano": "Ano"})
    st.plotly_chart(fig2, use_container_width=True)

# ── Análise por Fase ──────────────────────────────────────────────────────────
elif pagina == "Análise por Fase":
    st.title("Análise por Fase")

    fase_sel = st.selectbox("Selecione a fase", df["fase"].unique())
    ano_sel  = st.selectbox("Selecione o ano",  sorted(df["ano"].unique()))

    subset = df[(df["fase"] == fase_sel) & (df["ano"] == ano_sel)]
    st.write(f"{len(subset)} alunos em {fase_sel} — {ano_sel}")

    indicadores = ["ian", "ida", "ieg", "iaa", "ips", "ipp", "ipv", "inde"]
    fig = px.box(
        subset.melt(value_vars=indicadores, var_name="Indicador", value_name="Valor"),
        x="Indicador", y="Valor", color="Indicador",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(subset[["id_aluno"] + indicadores + ["em_risco"]], use_container_width=True)

# ── Previsão de Risco ─────────────────────────────────────────────────────────
elif pagina == "Previsão de Risco":
    st.title("Previsão de Risco de Defasagem")

    if model is None:
        st.warning("Modelo ainda não treinado. Execute o notebook de modelagem e salve em `models/risk_model.pkl`.")
    else:
        st.markdown("Insira os indicadores do aluno para estimar a probabilidade de risco.")

        c1, c2, c3, c4 = st.columns(4)
        ian = c1.slider("IAN", 0.0, 10.0, 5.0)
        ida = c2.slider("IDA", 0.0, 10.0, 5.0)
        ieg = c3.slider("IEG", 0.0, 10.0, 5.0)
        iaa = c4.slider("IAA", 0.0, 10.0, 5.0)

        c5, c6, c7 = st.columns(3)
        ips = c5.slider("IPS", 0.0, 10.0, 5.0)
        ipp = c6.slider("IPP", 0.0, 10.0, 5.0)
        ipv = c7.slider("IPV", 0.0, 10.0, 5.0)

        if st.button("Calcular risco"):
            entrada = pd.DataFrame([[ian, ida, ieg, iaa, ips, ipp, ipv]],
                                   columns=["ian", "ida", "ieg", "iaa", "ips", "ipp", "ipv"])
            prob = model.predict_proba(entrada)[0][1]

            if prob >= 0.7:
                st.error(f"⚠️ Alto risco: {prob:.1%}")
            elif prob >= 0.4:
                st.warning(f"⚡ Risco moderado: {prob:.1%}")
            else:
                st.success(f"✅ Baixo risco: {prob:.1%}")
