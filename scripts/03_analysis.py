"""
Exporta para outputs/ os agregados que sustentam as respostas às 11 perguntas do Datathon — as perguntas
sem função própria aqui são respondidas diretamente em notebooks/01_eda.ipynb, que também consome estes
CSVs. Executa após 02_transform.py (espera em_risco já calculado no painel).
"""

import os
import pandas as pd

INPUT_PATH = "../data/processed/pede_unificado.csv"


def p1_defasagem_ian(df: pd.DataFrame) -> pd.DataFrame:
    """P1 — Perfil de defasagem por ano e fase: IAN médio e contagens de defasagem severa (defas <= -2) e
    moderada (defas == -1), seguindo os cortes que o próprio IAN codifica."""
    g = df.groupby(["ano", "fase"])
    return g.agg(
        ian_medio=("ian", "mean"),
        defasagem_severa=("defas", lambda s: (s <= -2).sum()),
        defasagem_moderada=("defas", lambda s: (s == -1).sum()),
        total=("ian", "size"),
    ).reset_index()


def p2_desempenho_ida(df: pd.DataFrame) -> pd.DataFrame:
    """P2 — Evolução do desempenho acadêmico: IDA médio por ano e fase."""
    return df.groupby(["ano", "fase"])["ida"].mean().reset_index(name="ida_medio")


def p3_engajamento(df: pd.DataFrame) -> pd.DataFrame:
    """P3 — Engajamento vs. desempenho: painel aluno-ano com IEG, IDA e IPV, base das correlações e das
    trajetórias individuais exploradas no notebook de EDA."""
    return df[["id_aluno", "ano", "ieg", "ida", "ipv"]].sort_values(["id_aluno", "ano"])


def p8_multidimensional(df: pd.DataFrame) -> pd.DataFrame:
    """P8 — Visão multidimensional: indicadores parciais, INDE e em_risco restritos às linhas com INDE
    presente — sem avaliação consolidada não há combinação de indicadores a analisar."""
    cols = ["ida", "ieg", "ips", "ipp", "inde", "em_risco"]
    return df.loc[df["inde"].notna(), cols]


def p10_efetividade(df: pd.DataFrame) -> pd.DataFrame:
    """P10 — Efetividade do programa por fase ao longo dos anos: INDE e IPV médios e número de alunos,
    permitindo separar melhora real de mudança de composição das turmas."""
    return df.groupby(["fase", "ano"]).agg(
        inde_medio=("inde", "mean"),
        ipv_medio=("ipv", "mean"),
        n_alunos=("id_aluno", "size"),
    ).reset_index()


ANALYSES = {
    "p1_defasagem_ian": p1_defasagem_ian,
    "p2_desempenho_ida": p2_desempenho_ida,
    "p3_engajamento": p3_engajamento,
    "p8_multidimensional": p8_multidimensional,
    "p10_efetividade": p10_efetividade,
}


def main():
    os.makedirs("../outputs", exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    for nome, fn in ANALYSES.items():
        resultado = fn(df)
        resultado.to_csv(f"../outputs/{nome}.csv", index=False)
        print(f"  {nome}: {len(resultado)} linhas exportadas")


if __name__ == "__main__":
    main()
