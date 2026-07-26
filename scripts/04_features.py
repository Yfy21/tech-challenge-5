"""
Monta a base de features do modelo early-warning de defasagem (pergunta 9). Executa após 02_transform.py.
Saída: data/processed/base_risco.csv.

Desenho t→t+1 (decidido em plano_modelagem_analise.md §1.2, opção a): cada linha é um par aluno × anos
consecutivos — features observadas no ano t, alvo `em_risco` (defas < 0) observado em t+1. Não há
circularidade: defas e indicadores do ano t são preditores legítimos do risco FUTURO; o que não pode é
usar os do próprio ano-alvo.
"""

import pandas as pd

INPUT_PATH = "../data/processed/pede_unificado.csv"
OUTPUT_PATH = "../data/processed/base_risco.csv"

# ian fica de fora: é função determinística de defas (2.5/5/10), redundante.
FEATURES_NUM = [
    "defas", "inde", "ida", "ieg", "iaa", "ips", "ipp", "ipv",
    "fase", "idade", "tempo_vinculo",
]
FEATURES_TREND = ["d_inde", "d_ieg", "d_ida"]  # variação t-1 → t (NaN no 1º ano)
FEATURES_CAT = ["genero", "tipo_instituicao"]


def normalizar_tipo_instituicao(v) -> str:
    """Harmoniza os rótulos de instituição de ensino, que mudam a cada ano do PEDE: 2022 traz 'Escola
    Pública', 'Rede Decisão' e 'Escola JP II'; 2023/24 trazem 'Pública', variações de 'Privada ...' e
    situações especiais ('Concluiu o 3º EM', 'Bolsista Universitário', 'Nenhuma das opções'). Tudo é
    reduzido a três categorias estáveis (Pública/Privada/Outra) para o one-hot não fragmentar por
    grafia."""
    if pd.isna(v):
        return "Outra"
    s = str(v).lower()
    if "pública" in s or "publica" in s:
        return "Pública"
    if s.startswith("privada") or "rede decisão" in s or "jp ii" in s:
        return "Privada"
    return "Outra"


def montar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói os pares t→t+1: normaliza as categóricas, deriva as tendências d_inde/d_ieg/d_ida (valor
    do ano t menos o do ano t-1; NaN quando o aluno não tem o ano anterior) e junta o em_risco do ano
    seguinte como alvo. Ficam de fora os pares sem alvo definido (fase anômala '9' em t+1) e as linhas sem
    INDE no ano t — sem avaliação consolidada não há features que sustentem a predição."""
    df = df.copy()
    df["tipo_instituicao"] = df["tipo_instituicao"].map(normalizar_tipo_instituicao)
    df["em_risco"] = (
        df["em_risco"].map(lambda v: pd.NA if pd.isna(v) else v in (True, "True"))
        .astype("boolean")
    )

    # tendência: valor do ano t menos o do ano t-1 (quando o aluno tem t-1)
    ant = df[["id_aluno", "ano", "inde", "ieg", "ida"]].copy()
    ant["ano"] += 1
    df = df.merge(ant, on=["id_aluno", "ano"], how="left", suffixes=("", "_ant"))
    for c in ("inde", "ieg", "ida"):
        df[f"d_{c}"] = df[c] - df[f"{c}_ant"]

    # alvo: em_risco do ano seguinte
    alvo = df[["id_aluno", "ano", "em_risco"]].copy()
    alvo["ano"] -= 1
    base = df.merge(
        alvo.rename(columns={"em_risco": "alvo_defasagem_t1"}),
        on=["id_aluno", "ano"],
        how="inner",
    )

    base = base[base["alvo_defasagem_t1"].notna()]
    # sem avaliação no ano t (INDE ausente) não há features — fora da base
    base = base[base["inde"].notna()]

    cols = (
        ["id_aluno", "ano"] + FEATURES_NUM + FEATURES_TREND + FEATURES_CAT
        + ["alvo_defasagem_t1"]
    )
    return base[cols].reset_index(drop=True)


def main():
    df = pd.read_csv(INPUT_PATH)
    base = montar_base(df)
    base.to_csv(OUTPUT_PATH, index=False)

    alvo = base["alvo_defasagem_t1"].astype(bool)
    print(f"base_risco: {len(base)} pares t→t+1 | {base.id_aluno.nunique()} alunos")
    print(f"alvo positivo (defasado em t+1): {alvo.sum()} ({alvo.mean() * 100:.1f}%)")
    print("pares por ano t:")
    print(base.groupby("ano").size().to_string())


if __name__ == "__main__":
    main()
