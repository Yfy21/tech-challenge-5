"""
Este scritp monta a base de features do modelo de defasagem early-warning. Executa após o 02_eda.ipynb,
que é quem fornece a evidência para a escolha das features. Saída: data/processed/base_risco.csv.

Desenho t→t+1: cada linha acompanha um aluno de um ano para o seguinte; features são observadas no ano t e
relacionadas
ao alvo `defasado` (defasagem < 0) em t+1, possibilitando a predição do risco futuro.
"""

import pandas as pd

INPUT_PATH = "../data/processed/pede_unificado.csv"
OUTPUT_PATH = "../data/processed/base_risco.csv"

FEATURES_NUM = ["defasagem", "inde", "ida", "ieg", "iaa_val", "iaa_recusa", "ips", "ipp", "ipv",
                "fase", "idade", "tempo_vinculo",] # ian não é incluído por ser redundante com a defasagem
FEATURES_TREND = ["d_inde", "d_ieg", "d_ida", "d_ipv"]  # variação t-1 → t
FEATURES_CAT = ["genero", "tipo_instituicao"]


def montar_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói o painel t→t+1, derivando as tendências d_inde/d_ieg/d_ida/d_ipv (diferença entre os indicadores
    no ano t e t-1; NaN quando o aluno não tem o ano anterior) e juntando a coluna "defasado" do ano seguinte
    como alvo. Ficam de fora as linhas sem INDE no ano t — sem avaliação consolidada não há features que
    sustentem a predição.

    Os quatro indicadores com tendência saem da P9 do 02_eda.ipynb, que mede a correlação de cada indicador
    do ano t com a defasagem em t+1: INDE (-0,34), IPV (-0,34), IEG (-0,22) e IDA (-0,21) são os de sinal
    mais forte. O IAA (0,02) é ruído. O IPS (+0,16) entra só em nível porque a sua escala não é comparável
    entre anos — a P5 mostra 28,0% dos alunos de 2023 no piso do indicador contra 0,1% em 2022, de modo que
    a diferença ano a ano mediria mudança de régua e não trajetória do aluno. O IAN é redundante com a
    defasagem e o IPP de 2022 é reconstruído algebricamente no 01_clean — a diferença 2022→2023 misturaria
    valor reconstruído com medido justamente no ano que fornece metade das transições.

    A autoavaliação entra em duas colunas, e não em uma. O `iaa` publicado traz zero onde o aluno se recusou
    a responder (P4 do EDA), e um zero desses não é autoimagem baixa — alimentá-lo ao modelo como nota seria
    ensinar uma queda de 8,6 pontos que nunca existiu. Então a nota entra como `iaa_val`, em que a recusa
    vira ausente e é imputada pela mediana no pipeline, e o comportamento entra separado em `iaa_recusa`,
    que o modelo pode usar como sinal por conta própria: em 2022 quem recusou evadiu 51,3% contra 29,2% de
    quem respondeu (p = 0,006). É a informação da recusa preservada sem contaminar a escala do indicador.
    """

    # binária como float: o CSV intermediário perderia o dtype booleano do pandas, e o pipeline do modelo
    # trata a coluna no mesmo bloco numérico das demais (mediana + escala)
    df["iaa_recusa"] = df["iaa_recusa"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    # tendência: valor do ano t menos o do ano t-1 (quando o aluno tem t-1)
    ant = df[["id_aluno", "ano", "inde", "ieg", "ida", "ipv"]].copy()
    ant["ano"] += 1
    df = df.merge(ant, on=["id_aluno", "ano"], how="left", suffixes=("", "_ant"))
    for c in ("inde", "ieg", "ida", "ipv"):
        df[f"d_{c}"] = df[c] - df[f"{c}_ant"]

    # alvo: defasado do ano seguinte
    alvo = df[["id_aluno", "ano", "defasado"]].copy()
    alvo["ano"] -= 1
    base = df.merge(
        alvo.rename(columns={"defasado": "alvo_defasagem_t1"}),
        on=["id_aluno", "ano"],
        how="inner",
    )

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
    print(f"base_risco: {len(base)} transições t→t+1 | {base.id_aluno.nunique()} alunos")
    print(f"alvo positivo (defasado em t+1): {alvo.sum()} ({alvo.mean() * 100:.1f}%)")
    print("transições por ano t:")
    print(base.groupby("ano").size().to_string())


if __name__ == "__main__":
    main()
