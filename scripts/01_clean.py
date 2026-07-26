"""
Harmoniza as 3 planilhas do dataset bruto (PEDE2022/2023/2024 — cada uma com estrutura de colunas própria)
em um único dataset em formato longo (painel), com uma linha por (id_aluno, ano).

Produz dois arquivos: data/processed/pede_unificado.csv, o painel principal com os indicadores por ano, e
data/processed/evasao.csv, com as transições ano-a-ano por aluno (evadiu = presente no ano N e ausente no
ano N+1).

Decisões de harmonização (ver Obsidian: "Plano de Ação Datathon" e "PEDE Associação Passos Mágicos", além
da discussão com André em 2026-07-19):
  - Os indicadores estão na escala 0-10, não na 0-1.
  - O IPP ausente em 2022 é reconstruído algebricamente a partir do INDE (cujos pesos são conhecidos), em
    vez de imputado estatisticamente.
  - `Atingiu PV` e `Defas` são recriados a partir de IPV/Fase — não se confia no dado bruto, pois ambos
    estão sinalizados como propensos a erro nos documentos-fonte.
  - `Turma`, `Nome` e `Cg` são descartados por serem redundantes ou ruidosos.
  - `INDE 23` e `Pedra 23`, na planilha 2023, são colunas remanescentes quase inteiramente vazias (~8% de
    coincidência com `INDE 2023`/`Pedra 2023`) — descartadas em favor das colunas com o ano por extenso.
  - `Ativo/Inativo` (2024) é constante ("Cursando" em 100% das linhas) e foi descartada; a evasão é
    calculada via diferença de conjuntos de id_aluno entre anos (evasao.csv), não via essa coluna.
  - As colunas de texto livre (Rec Av n, Rec Psicologia, Destaque IEG/IDA/IPV) só existem preenchidas em
    2022 — ficam no painel para uma análise de texto à parte, mas não devem alimentar o modelo tabular
    principal.
"""

import re

import numpy as np
import pandas as pd

RAW_PATH = "../data/raw/dataset.xlsx"
OUTPUT_PATH = "../data/processed/pede_unificado.csv"
EVASAO_PATH = "../data/processed/evasao.csv"

GENERO_MAP = {"Menino": 1, "Masculino": 1, "Menina": 0, "Feminino": 0}

# Pesos oficiais do INDE (fases 0-7). Fase 8 usa outro esquema (IPP/IPV N/A) e não é usada na reconstrução
# do IPP porque a planilha 2022 não contém alunos com Fase == 8 (o valor máximo observado na coluna Fase é 7).
INDE_WEIGHTS_0_7 = {"ian": 0.1, "ida": 0.2, "ieg": 0.2, "iaa": 0.1, "ips": 0.1, "ipp": 0.1, "ipv": 0.2}

# Colunas do bloco "ano corrente" que cada planilha bruta fornece, mapeadas para o esquema padrão do painel.
# Colunas ausentes em um ano viram NaN após o concat (ex.: ipp em 2022, escola fora de 2024).
MAPA_2022 = {
    "ra": "id_aluno",
    "fase": "fase_raw",
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
    "matem": "matem",
    "portug": "portug",
    "inglês": "ingles",
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
    "mat": "matem",
    "por": "portug",
    "ing": "ingles",
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
    "mat": "matem",
    "por": "portug",
    "ing": "ingles",
    "indicado": "indicado_bolsa", "ipv": "ipv", "ian": "ian",
    "fase ideal": "fase_ideal_raw",
    "destaque ieg": "destaque_ieg",
    "destaque ida": "destaque_ida",
    "destaque ipv": "destaque_ipv",
    "data de nasc": "data_nasc",
    "escola": "escola",
}


def _extrair_fase_num(valor) -> float:
    """Normaliza formatos distintos de 'Fase'/'Fase ideal' por ano (int, 'ALFA'/'FASE 8', '8A',
    'Fase 8 (Universitários)') para um inteiro 0-8. Usa re.search (não re.match) porque o dígito nem sempre
    está no início da string (ex.: 'Fase 8 (...)'). Valores fora do intervalo 0-8 (ex.: o '9' anômalo em 2024)
    viram NaN em vez de serem silenciosamente aceitos."""
    s = str(valor).strip().upper()
    if s.startswith("ALFA"):
        return 0
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    n = int(m.group(1))
    return float(n) if 0 <= n <= 8 else np.nan


def _carregar_ano(sheet: pd.DataFrame, ano: int, mapa: dict) -> pd.DataFrame:
    sheet = sheet.copy()
    sheet.columns = sheet.columns.str.lower().str.strip()
    df = sheet[list(mapa.keys())].rename(columns=mapa)
    df["ano"] = ano
    return df


def carregar_e_harmonizar() -> pd.DataFrame:
    planilhas = pd.read_excel(RAW_PATH, sheet_name=[0, 1, 2])
    df22, df23, df24 = planilhas.values()

    partes = [
        _carregar_ano(df22, 2022, MAPA_2022),
        _carregar_ano(df23, 2023, MAPA_2023),
        _carregar_ano(df24, 2024, MAPA_2024),
    ]
    df = pd.concat(partes, ignore_index=True, sort=False)

    df["id_aluno"] = df["id_aluno"].astype(str).str.replace(r"^RA-", "", regex=True)
    # 'INDE' traz o literal 'INCLUIR' em 38 linhas de 2024 (alunos novos ainda sem nota consolidada) —
    # vira NaN em vez de quebrar o dtype.
    df["inde"] = pd.to_numeric(df["inde"], errors="coerce")
    df["genero"] = df["genero"].map(GENERO_MAP)
    df["fase"] = df["fase_raw"].apply(_extrair_fase_num)
    df["fase_ideal"] = df["fase_ideal_raw"].apply(_extrair_fase_num)
    df["fase_grupo"] = np.where(df["fase"] == 8, "fase_8", "fase_0_7")
    df.drop(columns=["fase_raw", "fase_ideal_raw"], inplace=True)

    # Ano/idade de nascimento: unifica 'ano_nasc' (2022, já é ano) e 'data_nasc' (2023/2024, data completa) —
    # idade é recriada a partir disso em vez de confiar nas colunas "Idade 22"/"Idade" do bruto.
    if "data_nasc" in df.columns:
        ano_nasc_2023_24 = pd.to_datetime(df["data_nasc"], errors="coerce").dt.year
        df["ano_nasc"] = df["ano_nasc"].fillna(ano_nasc_2023_24)
        df.drop(columns=["data_nasc"], inplace=True)
    df["idade"] = df["ano"] - df["ano_nasc"]
    df["tempo_vinculo"] = df["ano"] - df["ano_ingresso"]

    # Defasagem recriada a partir de fase atual vs. fase ideal (não confia na coluna 'Defas'/'Defasagem'
    # do bruto, sinalizada como propensa a erro).
    df["defas"] = df["fase"] - df["fase_ideal"]

    df = _reconstruir_ipp(df)
    df = _recriar_atingiu_pv(df)

    df = df.sort_values(["id_aluno", "ano"]).reset_index(drop=True)
    return df


def _reconstruir_ipp(df: pd.DataFrame) -> pd.DataFrame:
    """IPP não existe em 2022. Como o INDE é uma média ponderada com pesos conhecidos e os demais 6 indicadores
    estão presentes, o IPP é isolado algebricamente em vez de imputado. Só é válido para fase 0-7 (fase 8 não
    usa IPP no cálculo do INDE) — não se aplica a 2022, pois nenhuma linha daquele ano tem fase == 8."""
    falta_ipp = df["ipp"].isna() & (df["fase_grupo"] == "fase_0_7")

    outros = sum(
        INDE_WEIGHTS_0_7[k] * df[k]
        for k in ("ian", "ida", "ieg", "iaa", "ips", "ipv")
    )
    ipp_reconstruido = (df["inde"] - outros) / INDE_WEIGHTS_0_7["ipp"]

    df.loc[falta_ipp, "ipp"] = ipp_reconstruido[falta_ipp]
    return df


def _recriar_atingiu_pv(df: pd.DataFrame) -> pd.DataFrame:
    """Recria 'Atingiu PV' a partir da própria regra da associação (IPV >= média + 1 desvio padrão), calculada
    por ano — não confia na coluna bruta 'Atingiu PV', sinalizada como propensa a erro."""
    limiar = df.groupby("ano")["ipv"].transform(lambda s: s.mean() + s.std())
    df["atingiu_pv"] = df["ipv"] >= limiar
    return df


def construir_tabela_evasao(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada par de anos consecutivos, marca quais id_aluno presentes no ano N desaparecem no ano N+1
    (evadiu=True). Substitui o método de diferença de coorte por idade usado no relatório oficial e a coluna
    'Ativo/Inativo' (constante, não informativa)."""
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
    df = carregar_e_harmonizar()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Painel unificado: {len(df)} linhas ({df['id_aluno'].nunique()} alunos distintos) -> {OUTPUT_PATH}")

    evasao = construir_tabela_evasao(df)
    evasao.to_csv(EVASAO_PATH, index=False)
    taxa = evasao.groupby("ano_referencia")["evadiu"].mean()
    print(f"Tabela de evasão: {len(evasao)} transições -> {EVASAO_PATH}")
    print(f"Taxa de evasão por ano de referência:\n{taxa}")


if __name__ == "__main__":
    main()
