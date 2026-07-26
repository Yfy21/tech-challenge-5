"""
Correções semânticas e marcação de risco (em_risco) sobre o painel já harmonizado. Executa após 01_clean.py
ter produzido data/processed/pede_unificado.csv (indicadores na escala 0-10, com `defas` recriado a partir
de fase vs. fase ideal) e reescreve o mesmo arquivo.

em_risco != atingiu_pv: em_risco responde à pergunta 9 do datathon ("risco de defasagem"); atingiu_pv é o
indicador de ponto de virada já calculado em 01_clean.py — são alvos distintos, não devem ser confundidos.
"""

import pandas as pd

INPUT_PATH = "../data/processed/pede_unificado.csv"
OUTPUT_PATH = "../data/processed/pede_unificado.csv"


def calcular_risco(df: pd.DataFrame) -> pd.Series:
    """Marca como em risco o aluno defasado: defas < 0 (fase abaixo da fase ideal), o que equivale a
    ian <= 5 — o IAN é função determinística da defasagem (2.5 = severa, 5 = moderada, 10 = em fase). A
    definição é fixa e comparável entre anos, e substitui (2026-07-26) a heurística anterior (defas > 0 |
    ian < média-desvio do ano), que tinha o sinal de defas invertido (defas positivo = ADIANTADO) e um
    limiar adaptativo que só cruzava o valor discreto 5.0 em 2024, inflando artificialmente a taxa de
    risco de ~5% para ~55%. Linhas com defas ausente (fase anômala '9' de 2024) ficam com <NA> e devem
    ser excluídas da modelagem."""
    return (df["defas"] < 0).mask(df["defas"].isna())


def corrigir_ieg_fase8(df: pd.DataFrame) -> pd.DataFrame:
    """A fase 8 (universitários) é avaliada por rubrica própria: no painel não há IDA/IAA/IPS/IPP/IPV/INDE
    para essas linhas. Em 2024, porém, o IEG veio preenchido com 0.0 nas 64 linhas de fase 8 (em 2023 vem
    vazio) — zeros-placeholder, não engajamento real. Sem a correção, qualquer agregado de IEG por ano ou
    fase seria distorcido (o IEG médio de 2024 cairia artificialmente de 7.81 para ~7.36)."""
    df.loc[df["fase"] == 8, "ieg"] = pd.NA
    return df


def main():
    df = pd.read_csv(INPUT_PATH)

    df = corrigir_ieg_fase8(df)
    df["em_risco"] = calcular_risco(df)

    df.to_csv(OUTPUT_PATH, index=False)

    n_risco = df["em_risco"].sum()
    print(f"Marcação concluída: {n_risco} avaliações em risco de {len(df)}.")


if __name__ == "__main__":
    main()
