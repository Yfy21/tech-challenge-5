"""
Este script unifica e harmoniza as 3 planilhas do dataset bruto (PEDE2022/2023/2024 — cada uma com uma
estrutura própria de colunas) em um único dataset de formato longo, com id composto (id_aluno, ano).

Além desse conjunto de dados principal, armazenado em data/processed/pede_unificado.csv, também é construído
o arquivo data/processed/evasao.csv, com as transições ano-a-ano por aluno, o que nos permite analisar a
evasão (evadiu = presente no ano N e ausente no ano N+1).
"""

import re

import numpy as np
import pandas as pd

RAW_PATH = "../data/raw/dataset.xlsx"
OUTPUT_PATH = "../data/processed/pede_unificado.csv"
EVASAO_PATH = "../data/processed/evasao.csv"

# Dicionário de normalização do campo "gênero"
GENERO_MAP = {"Menino": "Masc", "Masculino": "Masc", "Menina": "Fem", "Feminino": "Fem"}

# Dicionário de normalização das pedras. Qualquer outro valor vira NaN, como o placeholder "INCLUIR",
# que aparece em 2024 nas linhas sem INDE calculado.
PEDRA_MAP = {
    "quartzo": "Quartzo",
    "agata": "Ágata",
    "ágata": "Ágata",
    "ametista": "Ametista",
    "topazio": "Topázio",
    "topázio": "Topázio",
}

# Pesos oficiais do INDE para as fases de 0 a 7, apresentados nos relatórios de pesquisa disponibilizados
# junto ao dataset. Este dicionário é utilizado para reconstruir os valores do IPP, ausentes no ano de 2022.
# Como não há alunos da fase 8 em 2022, não é necessário um dicionário com os pesos dessa fase, que diferem
# das demais.
INDE_WEIGHTS_0_7 = {"ian": 0.1, "ida": 0.2, "ieg": 0.2, "iaa": 0.1, "ips": 0.1, "ipp": 0.1, "ipv": 0.2}

# Dicionários para normalização das colunas.
MAPA_2022 = {
    "ra": "id_aluno",
    "fase": "fase_raw", # Requer transformações futuras
    "gênero": "genero",
    "ano nasc": "ano_nasc",
    "ano ingresso": "ano_ingresso",
    "instituição de ensino": "tipo_instituicao",
    "pedra 22": "pedra",
    "inde 22": "inde",
    "cf": "cf",
    "ct": "ct",
    "nº av": "n_av",
    "avaliador1": "avaliador_1",
    "rec av1": "rec_av_1",
    "avaliador2": "avaliador_2",
    "rec av2": "rec_av_2",
    "avaliador3": "avaliador_3",
    "rec av3": "rec_av_3",
    "avaliador4": "avaliador_4",
    "rec av4": "rec_av_4",
    "iaa": "iaa",
    "ieg": "ieg",
    "ips": "ips",
    "rec psicologia": "rec_psicologia",
    "ida": "ida",
    "matem": "mat",
    "portug": "port",
    "inglês": "ing",
    "indicado": "indicado_bolsa",
    "ipv": "ipv",
    "ian": "ian",
    "fase ideal": "fase_ideal_raw",
    "destaque ieg": "destaque_ieg",
    "destaque ida": "destaque_ida",
    "destaque ipv": "destaque_ipv",
}

MAPA_2023 = {
    "ra": "id_aluno",
    "fase": "fase_raw",
    "gênero": "genero",
    "ano ingresso": "ano_ingresso",
    "instituição de ensino": "tipo_instituicao",
    "pedra 2023": "pedra",
    "inde 2023": "inde",
    "cf": "cf",
    "ct": "ct",
    "nº av": "n_av",
    "avaliador1": "avaliador_1",
    "rec av1": "rec_av_1",
    "avaliador2": "avaliador_2",
    "rec av2": "rec_av_2",
    "avaliador3": "avaliador_3",
    "rec av3": "rec_av_3",
    "avaliador4": "avaliador_4",
    "rec av4": "rec_av_4",
    "iaa": "iaa",
    "ieg": "ieg",
    "ips": "ips",
    "ipp": "ipp",
    "rec psicologia": "rec_psicologia",
    "ida": "ida",
    "mat": "mat",
    "por": "port",
    "ing": "ing",
    "indicado": "indicado_bolsa",
    "ipv": "ipv",
    "ian": "ian",
    "fase ideal": "fase_ideal_raw",
    "destaque ieg": "destaque_ieg",
    "destaque ida": "destaque_ida",
    "destaque ipv": "destaque_ipv",
    "data de nasc": "data_nasc",
}

MAPA_2024 = {
    "ra": "id_aluno",
    "fase": "fase_raw",
    "gênero": "genero",
    "ano ingresso": "ano_ingresso",
    "instituição de ensino": "tipo_instituicao",
    "pedra 2024": "pedra",
    "inde 2024": "inde",
    "cf": "cf",
    "ct": "ct",
    "nº av": "n_av",
    "avaliador1": "avaliador_1",
    "rec av1": "rec_av_1",
    "avaliador2": "avaliador_2",
    "rec av2": "rec_av_2",
    "avaliador3": "avaliador_3",
    "avaliador4": "avaliador_4",
    "avaliador5": "avaliador_5",
    "avaliador6": "avaliador_6",
    "iaa": "iaa",
    "ieg": "ieg",
    "ips": "ips",
    "ipp": "ipp",
    "rec psicologia": "rec_psicologia",
    "ida": "ida",
    "mat": "mat",
    "por": "port",
    "ing": "ing",
    "indicado": "indicado_bolsa", "ipv": "ipv", "ian": "ian",
    "fase ideal": "fase_ideal_raw",
    "destaque ieg": "destaque_ieg",
    "destaque ida": "destaque_ida",
    "destaque ipv": "destaque_ipv",
    "data de nasc": "data_nasc",
    "escola": "escola",
}


# Definição das Funções

def _carregar_ano(sheet: pd.DataFrame, ano: int, mapa: dict) -> pd.DataFrame:
    """
    Adiciona uma coluna de ano a cada df para possibilitar a estruturação da tabela de formato longo.
    """
    sheet = sheet.copy()
    sheet.columns = sheet.columns.str.lower().str.strip()
    df = sheet[list(mapa.keys())].rename(columns=mapa)
    df["ano"] = ano
    return df


def _extrair_fase_num(valor) -> float:
    """
    Normaliza formatos distintos de 'Fase' e 'Fase ideal' por ano (int, 'ALFA'/'FASE 8', '8A',
    'Fase 8 (Universitários)') para um inteiro 0-8. Valores fora do intervalo 0-8 (ex.: o '9' anômalo em
    2024) viram NaN.
    """
    s = str(valor).strip().upper()
    if s.startswith("ALFA"):
        return 0
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    n = int(m.group(1))
    return float(n) if 0 <= n <= 8 else np.nan


def _normalizar_tipo_instituicao(v):
    """
    Harmoniza os tipos de instituições de ensino em "Pública" ou "Privada"; situações que não informam o tipo
    de instituição ('Concluiu o 3º EM', 'Nenhuma das opções') viram NaN, como os ausentes.
    """
    if pd.isna(v):
        return np.nan
    s = str(v).lower()
    if "pública" in s or "publica" in s:
        return "Pública"
    if s.startswith("privada") or "rede decisão" in s or "jp ii" in s or "bolsista" in s:
        return "Privada"
    return np.nan


def _marcar_recusa_iaa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Como o próprio relatório de 2022 afirma no início da página 117: 'No indicador de autoavaliação a nota
    ZERO significa a não participação do estudante, por escolha própria, nesse processo de autoavaliação'.
    Isso significa que os zeros dessa coluna não devem ser encarados como uma nota como outra qualquer, mas
    sim como um valor booleano de não participação. Como a organização considera os zeros na construção do
    INDE e, consequentemente, na categorização das pedras, não os descartei. No entanto, criei mais duas co-
    lunas que permitem uma análise mais detalhada do assunto.
    """
    df["iaa_recusa"] = df["iaa"].eq(0).astype("boolean").mask(df["iaa"].isna())
    df["iaa_val"] = df["iaa"].replace(0, np.nan)
    return df


def _reconstruir_ipp(df: pd.DataFrame) -> pd.DataFrame:
    """
    O IPP não existe em 2022. Como o INDE é uma média ponderada com pesos conhecidos e os demais 6
    indicadores estão presentes, esta função isola o IPP algebricamente para povoar os dados de 2022.
    A função é aplicada apenas nos alunos de fase 0 a 7 (ver comentário sobre isso em INDE_WEIGHTS_0_7).
    """
    falta_ipp = df["ipp"].isna() & (df["fase_grupo"] == "fase_0_7")

    outros = sum(
        INDE_WEIGHTS_0_7[k] * df[k]
        for k in ("ian", "ida", "ieg", "iaa", "ips", "ipv")
    )
    ipp_reconstruido = (df["inde"] - outros) / INDE_WEIGHTS_0_7["ipp"]

    df.loc[falta_ipp, "ipp"] = ipp_reconstruido[falta_ipp]
    return df


def _recriar_atingiu_pv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recria 'Atingiu PV' a partir da própria regra da associação, apresentada nos relatórios de pesquisa
    (IPV >= média + 1 desvio padrão), evitando confiar na coluna bruta 'Atingiu PV', que pode conter erros.
    """
    limiar = df.groupby("ano")["ipv"].transform(lambda s: s.mean() + s.std())
    df["atingiu_pv"] = df["ipv"] >= limiar
    return df


def _consolidar_ano_ingresso(df: pd.DataFrame) -> pd.DataFrame:
    """
    A planilha de 2024 não traz nenhum ingresso anterior a 2021, enquanto as de 2022 e 2023 chegam a 2016.
    Como o ano de ingresso é um fato imutável do aluno, essa divergência só pode ser perda de informação, e
    o registro mais antigo é o correto. Sem essa consolidação, o tempo de vínculo satura em 3 anos em 2024,
    justamente para os alunos de casa mais antiga.
    """
    df["ano_ingresso"] = df.groupby("id_aluno")["ano_ingresso"].transform("min")
    return df


def carregar_e_harmonizar() -> pd.DataFrame:
    """
    Função principal que aplica as anteriores.
    """
    planilhas = pd.read_excel(RAW_PATH, sheet_name=[0, 1, 2])
    df22, df23, df24 = planilhas.values()

    partes = [
        _carregar_ano(df22, 2022, MAPA_2022),
        _carregar_ano(df23, 2023, MAPA_2023),
        _carregar_ano(df24, 2024, MAPA_2024),
    ]
    df = pd.concat(partes, ignore_index=True, sort=False)

    df["id_aluno"] = df["id_aluno"].astype(str).str.replace(r"^RA-", "", regex=True)
    df["inde"] = pd.to_numeric(df["inde"], errors="coerce") # Transforma em NaN os valores "INCLUIR" presentes em 2024
    df["genero"] = df["genero"].map(GENERO_MAP)
    df["pedra"] = df["pedra"].astype(str).str.strip().str.lower().map(PEDRA_MAP)
    df["tipo_instituicao"] = df["tipo_instituicao"].map(_normalizar_tipo_instituicao)
    df["fase"] = df["fase_raw"].apply(_extrair_fase_num)
    df["fase_ideal"] = df["fase_ideal_raw"].apply(_extrair_fase_num)
    df["fase_grupo"] = np.where(df["fase"] == 8, "fase_8", "fase_0_7")
    df.drop(columns=["fase_raw", "fase_ideal_raw"], inplace=True)

    # O if statement a seguir unifica 'ano_nasc' (em 2022, apenas ano) e 'data_nasc' (em 2023/2024, a data
    # completa). A linha que o sucede recria a idade a partir do ano de nascimento em vez de confiar nas
    # colunas "Idade 22" e "Idade" dos dados brutos.
    if "data_nasc" in df.columns:
        ano_nasc_2023_24 = pd.to_datetime(df["data_nasc"], errors="coerce").dt.year
        df["ano_nasc"] = df["ano_nasc"].fillna(ano_nasc_2023_24)
        df.drop(columns=["data_nasc"], inplace=True)
    df["idade"] = df["ano"] - df["ano_nasc"]
    df = _consolidar_ano_ingresso(df)
    df["tempo_vinculo"] = df["ano"] - df["ano_ingresso"]

    # A defasagem também é recriada matematicamente (fase atual - fase ideal) em vez de confiar na coluna
    # 'Defas'/'Defasagem' das planilhas originais, evitando possíveis erros.
    df["defasagem"] = df["fase"] - df["fase_ideal"]

    df = _marcar_recusa_iaa(df)
    df = _reconstruir_ipp(df)
    df = _recriar_atingiu_pv(df)

    # Em 2024, não temos dados sobre nenhum indicador dos estudantes universitários. No entanto, a coluna IEG
    # desses alunos possui zeros colocados como placeholders. A linha a seguir conserta isso.
    df.loc[(df["ano"] == 2024) & (df["fase"] == 8), "ieg"] = pd.NA
    df["defasado"] = df["defasagem"] < 0 # Cria a coluna que servirá de alvo para o modelo

    df = df.sort_values(["id_aluno", "ano"]).reset_index(drop=True)
    return df


def construir_tabela_evasao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada par de anos consecutivos, marca quais id_aluno presentes no ano N desaparecem no ano N+1
    (evadiu=True). Substitui o método de diferença de coorte por idade usado no relatório oficial e a coluna
    'Ativo/Inativo' (constante, não informativa).
    """
    anos = sorted(df["ano"].unique())
    linhas = []
    for ano_atual, ano_seguinte in zip(anos, anos[1:]):
        alunos_atual = set(df.loc[df["ano"] == ano_atual, "id_aluno"])
        alunos_seguinte = set(df.loc[df["ano"] == ano_seguinte, "id_aluno"])
        for id_aluno in alunos_atual:
            linhas.append({
                "id_aluno": id_aluno,
                "ano_referencia": ano_atual,
                "ano_seguinte": ano_seguinte,
                "evadiu": id_aluno not in alunos_seguinte,
            })
    return pd.DataFrame(linhas)


def main():
    """
    Executa as duas principais funções do script.
    """
    df = carregar_e_harmonizar()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Painel unificado: {len(df)} linhas ({df['id_aluno'].nunique()} alunos distintos) -> {OUTPUT_PATH}")
    print(f"Marcação de defasagem: {df['defasado'].sum()} avaliações defasadas de {len(df)}.")
    recusa = df.groupby("ano")["iaa_recusa"].agg(["sum", "mean"])
    print(f"Recusa de autoavaliação (IAA = 0) por ano:\n{recusa}")

    evasao = construir_tabela_evasao(df)
    evasao.to_csv(EVASAO_PATH, index=False)
    taxa = evasao.groupby("ano_referencia")["evadiu"].mean()
    print(f"Tabela de evasão: {len(evasao)} transições -> {EVASAO_PATH}")
    print(f"Taxa de evasão por ano de referência:\n{taxa}")


if __name__ == "__main__":
    main()
