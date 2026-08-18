"""
Análise estatística e incerteza Tipo A — Membrana M1
=====================================================

Fluxo da análise
-----------------
1. Verificação da autocorrelação dos dados.
2. O início do pós-transiente é especificado manualmente pelo usuário.
3. O tamanho da janela principal é especificado manualmente pelo usuário.
4. São mantidas as etapas de validação da escolha da janela:
   - ANOVA/Modelo Linear Geral das médias das sub-janelas;
   - diferença relativa entre médias das sub-janelas e a média da janela;
   - verificação de completude temporal.
5. Validação individual de cada janela antes de aplicar Zhang:
   - teste ADF;
   - presença de todas as sub-janelas;
   - completude temporal;
   - desvio máximo das médias das sub-janelas em relação à média da janela.
6. Cálculo da incerteza-padrão Tipo A pelo procedimento de Zhang (2006).

Configuração manual
-------------------
Altere INICIO_POS_TRANSIENTE_DIAS e TAMANHO_JANELA_HORAS no início do script.

As colunas Inicio_janela_dias e Fim_janela_dias informam os limites
teóricos de cada janela na escala original da coluna Tempo_dias.

Validações preservadas
----------------------
A ANOVA e os critérios de representatividade/completude permanecem no código.

Detalhes originais da validação de 24 h:
   - ANOVA/Modelo Linear Geral das médias das sub-janelas de 6 h;
   - diferença relativa entre médias de 6 h e a respectiva média de 24 h;
   - verificação de completude temporal das janelas.
4. Validação individual de cada janela de 24 h antes de aplicar Zhang:
   - teste ADF;
   - presença das quatro sub-janelas de 6 h;
   - completude temporal;
   - desvio máximo das médias de 6 h em relação à média de 24 h.
5. Cálculo da incerteza-padrão Tipo A pelo procedimento de Zhang (2006).

Referência metodológica principal
---------------------------------
ZHANG, Nien Fan. Calculation of the uncertainty of the mean of autocorrelated
measurements. Metrologia, v. 43, 2006, p. S276–S281.

Dependências
------------
pip install pandas numpy scipy statsmodels openpyxl
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

NOME_EXPERIMENTO = "M3"

ARQUIVO_DADOS = Path(
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\informações\dados_final_M3_preenchido.csv"
)

PASTA_SAIDA = Path(
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\informações"
)

SEP = "/"
DECIMAL = ","

COLUNA_TEMPO_DIAS = "Tempo_dias"
COLUNA_FLUXO = "fluxo_permeado"

# ------------------------------------------------------------
# DEFINIÇÕES MANUAIS
# ------------------------------------------------------------

# Início do pós-transiente, na escala original de Tempo_dias.
# Altere este valor após fazer sua análise da estabilidade da variabilidade.
INICIO_POS_TRANSIENTE_DIAS = 2.30

# Tamanho da janela principal usada na validação e em Zhang.
# Ex.: 24.0 para 24 h; 12.0 para 12 h.
TAMANHO_JANELA_HORAS = 12.0

# Tamanho da subjanela usada para verificar a representatividade
# da janela principal. Mantido em 6 h, como no procedimento original.
TAMANHO_SUBJANELA_HORAS = 6.0

# ------------------------------------------------------------
# Parâmetros das validações
# ------------------------------------------------------------

ALPHA_ANOVA = 0.05
ALPHA_ADF = 0.05

LIMITE_ERRO_MAX_PCT = 5.0
LIMITE_ERRO_P95_PCT = 5.0

# Fração mínima da duração teórica para considerar a janela completa.
COBERTURA_MINIMA = 0.95

# Número de lags para diagnóstico inicial de autocorrelação.
LAGS_DIAGNOSTICO = 500

# ------------------------------------------------------------
# Zhang (2006)
# ------------------------------------------------------------

Z_CRIT = 1.96

# ============================================================
# LEITURA E PREPARAÇÃO
# ============================================================

def carregar_dados() -> pd.DataFrame:
    print(f"1) Lendo os dados de {NOME_EXPERIMENTO}...")

    df = pd.read_csv(
        ARQUIVO_DADOS,
        sep=SEP,
        decimal=DECIMAL,
        low_memory=False,
    )

    df.columns = df.columns.str.strip()

    faltantes = [
        c for c in [COLUNA_TEMPO_DIAS, COLUNA_FLUXO]
        if c not in df.columns
    ]

    if faltantes:
        raise KeyError(
            f"Colunas ausentes: {faltantes}\n"
            f"Colunas encontradas: {df.columns.tolist()}"
        )

    df[COLUNA_TEMPO_DIAS] = pd.to_numeric(
        df[COLUNA_TEMPO_DIAS], errors="coerce"
    )
    df[COLUNA_FLUXO] = pd.to_numeric(
        df[COLUNA_FLUXO], errors="coerce"
    )

    df = (
        df.dropna(subset=[COLUNA_TEMPO_DIAS, COLUNA_FLUXO])
        .sort_values(COLUNA_TEMPO_DIAS)
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Nenhum dado válido após a leitura.")

    return df

# ============================================================
# ETAPA 1 — VERIFICAÇÃO DA AUTOCORRELAÇÃO
# ============================================================

def verificar_autocorrelacao(
    serie: np.ndarray,
    nome: str,
    nlags: int = LAGS_DIAGNOSTICO,
) -> tuple[dict, pd.DataFrame]:

    x = np.asarray(serie, dtype=float)
    x = x[np.isfinite(x)]

    N = len(x)

    if N < 3:
        raise ValueError(
            f"{nome}: dados insuficientes para ACF."
        )

    nlags = min(nlags, N - 1)

    rho = acf(
        x,
        nlags=nlags,
        fft=True,
        adjusted=False,
        missing="drop",
    )[1:]

    lags = np.arange(1, len(rho) + 1)

    # Limite aproximado convencional de 95% usado apenas
    # como diagnóstico inicial de presença de autocorrelação.
    limite = 1.96 / math.sqrt(N)

    significativo = np.abs(rho) > limite

    resumo = {
        "Serie": nome,
        "N": N,
        "Lags_avaliados": len(rho),
        "Limite_95_aprox": limite,
        "FAC_lag1": rho[0] if len(rho) else np.nan,
        "Qtd_lags_significativos": int(significativo.sum()),
        "Percentual_lags_significativos": (
            100 * significativo.mean()
            if len(significativo)
            else np.nan
        ),
        "Autocorrelacionada": bool(significativo.any()),
    }

    tabela = pd.DataFrame({
        "Lag": lags,
        "FAC": rho,
        "Abs_FAC": np.abs(rho),
        "Limite_95_aprox": limite,
        "Significativo": significativo.astype(int),
    })

    return resumo, tabela


# ============================================================
# ETAPA 2 — RECORTE MANUAL DO PÓS-TRANSIENTE E CRIAÇÃO DAS JANELAS
# ============================================================

def separar_pos_transiente(df: pd.DataFrame) -> pd.DataFrame:

    pos = df.loc[
        df[COLUNA_TEMPO_DIAS] >= INICIO_POS_TRANSIENTE_DIAS
    ].copy()

    if pos.empty:
        raise ValueError(
            f"Não há dados a partir de {INICIO_POS_TRANSIENTE_DIAS} dias."
        )

    # Tempo relativo ao início escolhido manualmente.
    pos["Tempo_pos_dias"] = (
        pos[COLUNA_TEMPO_DIAS] - INICIO_POS_TRANSIENTE_DIAS
    )

    janela_dias = TAMANHO_JANELA_HORAS / 24.0
    subjanela_dias = TAMANHO_SUBJANELA_HORAS / 24.0

    if janela_dias <= 0 or subjanela_dias <= 0:
        raise ValueError("Os tamanhos das janelas devem ser positivos.")

    razao = TAMANHO_JANELA_HORAS / TAMANHO_SUBJANELA_HORAS
    n_subjanelas = int(round(razao))

    if not np.isclose(razao, n_subjanelas):
        raise ValueError(
            "TAMANHO_JANELA_HORAS deve ser múltiplo de "
            "TAMANHO_SUBJANELA_HORAS para a validação por sub-janelas."
        )

    pos["Janela"] = (
        np.floor(pos["Tempo_pos_dias"] / janela_dias).astype(int) + 1
    )

    pos["Subjanela_global"] = (
        np.floor(pos["Tempo_pos_dias"] / subjanela_dias).astype(int) + 1
    )

    pos["Posicao_subjanela"] = (
        (pos["Subjanela_global"] - 1) % n_subjanelas
    ) + 1

    return pos


def limites_janela(janela: int) -> dict:
    """
    Retorna os limites teóricos da janela na escala original de Tempo_dias
    e na escala relativa ao início do pós-transiente.
    """
    janela_dias = TAMANHO_JANELA_HORAS / 24.0

    inicio_rel = (janela - 1) * janela_dias
    fim_rel = janela * janela_dias

    return {
        "Inicio_janela_dias": INICIO_POS_TRANSIENTE_DIAS + inicio_rel,
        "Fim_janela_dias": INICIO_POS_TRANSIENTE_DIAS + fim_rel,
        "Inicio_janela_pos_dias": inicio_rel,
        "Fim_janela_pos_dias": fim_rel,
    }

# ============================================================
# ETAPA 3 — VALIDAÇÃO DO TAMANHO DA JANELA
# ============================================================

def estatisticas_subjanelas(pos: pd.DataFrame) -> pd.DataFrame:

    return (
        pos.groupby(["Janela", "Posicao_subjanela"])[COLUNA_FLUXO]
        .agg(
            N="count",
            Media_sub="mean",
            DesvPad_sub="std",
            Minimo_sub="min",
            Q1_sub=lambda s: s.quantile(0.25),
            Mediana_sub="median",
            Q3_sub=lambda s: s.quantile(0.75),
            Maximo_sub="max",
        )
        .reset_index()
    )


def estatisticas_janelas(pos: pd.DataFrame) -> pd.DataFrame:

    janela_dias = TAMANHO_JANELA_HORAS / 24.0
    linhas = []

    for janela, sub in pos.groupby("Janela", sort=True):
        janela = int(janela)
        limites = limites_janela(janela)

        primeiro = float(sub[COLUNA_TEMPO_DIAS].min())
        ultimo = float(sub[COLUNA_TEMPO_DIAS].max())
        cobertura_dias = ultimo - primeiro

        linhas.append({
            "Janela": janela,
            **limites,
            "Primeiro_dado_dias": primeiro,
            "Ultimo_dado_dias": ultimo,
            "Cobertura_dias": cobertura_dias,
            "Fracao_janela": cobertura_dias / janela_dias,
            "Janela_completa": bool(
                cobertura_dias / janela_dias >= COBERTURA_MINIMA
            ),
            "N": int(sub[COLUNA_FLUXO].count()),
            "Media_janela": float(sub[COLUNA_FLUXO].mean()),
            "DesvPad_janela": float(sub[COLUNA_FLUXO].std()),
        })

    return pd.DataFrame(linhas)


def validar_representatividade(
    stats_sub: pd.DataFrame,
    stats_jan: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    comp = stats_sub.merge(
        stats_jan[["Janela", "Media_janela", "Janela_completa"]],
        on="Janela",
        how="left",
    )

    comp["Erro_rel_abs_pct"] = (
        np.abs(comp["Media_sub"] - comp["Media_janela"])
        / np.abs(comp["Media_janela"])
        * 100
    )

    erro_max = float(comp["Erro_rel_abs_pct"].max())
    erro_p95 = float(comp["Erro_rel_abs_pct"].quantile(0.95))

    resumo = {
        "Erro_max_pct": erro_max,
        "Erro_P95_pct": erro_p95,
        "Limite_erro_max_pct": LIMITE_ERRO_MAX_PCT,
        "Limite_erro_P95_pct": LIMITE_ERRO_P95_PCT,
        "Aprovada_erro_max": erro_max <= LIMITE_ERRO_MAX_PCT,
        "Aprovada_erro_P95": erro_p95 <= LIMITE_ERRO_P95_PCT,
    }

    resumo["Representatividade_aprovada"] = bool(
        resumo["Aprovada_erro_max"]
        and resumo["Aprovada_erro_P95"]
    )

    return comp, resumo


def anova_posicao_subjanela(
    stats_sub: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:

    dados = stats_sub.dropna(subset=["Media_sub"]).copy()
    dados["Posicao_subjanela"] = dados["Posicao_subjanela"].astype("category")
    dados["Janela"] = dados["Janela"].astype("category")

    modelo = smf.ols(
        "Media_sub ~ C(Posicao_subjanela) + C(Janela)",
        data=dados,
    ).fit()

    tabela = anova_lm(modelo, typ=2).reset_index()
    tabela = tabela.rename(columns={"index": "Fonte"})

    def valor_fonte(fonte: str, coluna: str):
        linha = tabela.loc[tabela["Fonte"] == fonte, coluna]
        return float(linha.iloc[0]) if not linha.empty else np.nan

    F_pos = valor_fonte("C(Posicao_subjanela)", "F")
    p_pos = valor_fonte("C(Posicao_subjanela)", "PR(>F)")
    F_jan = valor_fonte("C(Janela)", "F")
    p_jan = valor_fonte("C(Janela)", "PR(>F)")

    resumo = {
        "F_Posicao_subjanela": F_pos,
        "p_Posicao_subjanela": p_pos,
        "F_Janela": F_jan,
        "p_Janela": p_jan,
        "Sem_efeito_sistematico_posicao": bool(
            p_pos >= ALPHA_ANOVA
        ) if np.isfinite(p_pos) else False,
    }

    return tabela, resumo

# ============================================================
# ETAPA 4 — VALIDAÇÃO INDIVIDUAL DAS JANELAS DE 24 h
# ============================================================

def teste_adf(
    serie: np.ndarray
) -> dict:

    x = np.asarray(
        serie,
        dtype=float,
    )

    x = x[np.isfinite(x)]

    if len(x) < 20:
        return {
            "ADF": np.nan,
            "ADF_p": np.nan,
            "ADF_lags": np.nan,
            "ADF_nobs": np.nan,
            "ADF_estacionaria": False,
            "ADF_erro": "N insuficiente",
        }

    try:
        resultado = adfuller(
            x,
            regression="c",
            autolag="AIC",
        )

        return {
            "ADF": resultado[0],
            "ADF_p": resultado[1],
            "ADF_lags": resultado[2],
            "ADF_nobs": resultado[3],
            "ADF_estacionaria":
                bool(resultado[1] < ALPHA_ADF),
            "ADF_erro": "",
        }

    except Exception as exc:
        return {
            "ADF": np.nan,
            "ADF_p": np.nan,
            "ADF_lags": np.nan,
            "ADF_nobs": np.nan,
            "ADF_estacionaria": False,
            "ADF_erro": str(exc),
        }


def validar_janela(
    janela: int,
    sub: pd.DataFrame,
    stats_sub_janela: pd.DataFrame,
) -> dict:

    serie = sub[COLUNA_FLUXO].dropna().to_numpy(dtype=float)
    adf = teste_adf(serie)

    n_sub_esperadas = int(
        round(TAMANHO_JANELA_HORAS / TAMANHO_SUBJANELA_HORAS)
    )

    posicoes_presentes = sorted(
        stats_sub_janela["Posicao_subjanela"].unique().tolist()
    )

    todas_subjanelas = (
        posicoes_presentes == list(range(1, n_sub_esperadas + 1))
    )

    limites = limites_janela(janela)

    primeiro = float(sub[COLUNA_TEMPO_DIAS].min())
    ultimo = float(sub[COLUNA_TEMPO_DIAS].max())
    cobertura = ultimo - primeiro
    janela_dias = TAMANHO_JANELA_HORAS / 24.0

    completa = bool(
        cobertura / janela_dias >= COBERTURA_MINIMA
    )

    media_janela = float(np.mean(serie))

    erros_sub = (
        np.abs(stats_sub_janela["Media_sub"] - media_janela)
        / abs(media_janela)
        * 100
    )

    erro_max_sub = (
        float(erros_sub.max()) if len(erros_sub) else np.nan
    )

    erro_local_ok = bool(
        np.isfinite(erro_max_sub)
        and erro_max_sub <= LIMITE_ERRO_MAX_PCT
    )

    valida = bool(
        adf["ADF_estacionaria"]
        and todas_subjanelas
        and completa
        and erro_local_ok
    )

    return {
        **limites,
        "Primeiro_dado_dias": primeiro,
        "Ultimo_dado_dias": ultimo,
        **adf,
        "Todas_subjanelas_presentes": todas_subjanelas,
        "Cobertura_janela_dias": cobertura,
        "Janela_completa": completa,
        "Erro_max_sub_vs_janela_pct": erro_max_sub,
        "Erro_local_aprovado": erro_local_ok,
        "Valida_para_Zhang": valida,
    }

# ============================================================
# ETAPA 5 — INCERTEZA TIPO A SEGUNDO ZHANG
# ============================================================

def zhang_incerteza(
    serie: np.ndarray
) -> tuple[dict, pd.DataFrame]:

    x = np.asarray(
        serie,
        dtype=float,
    )

    x = x[np.isfinite(x)]

    N = len(x)

    if N < 4:
        raise ValueError(
            "N insuficiente para Zhang."
        )

    media = float(np.mean(x))

    s = float(
        np.std(x, ddof=1)
    )

    ep_indep = (
        s / math.sqrt(N)
    )

    max_lag = int(
        math.floor(N / 4)
    )

    rho_all = acf(
        x,
        nlags=max_lag,
        fft=True,
        adjusted=False,
        missing="drop",
    )

    rho = np.asarray(
        rho_all[1:max_lag + 1],
        dtype=float,
    )

    lags = np.arange(
        1,
        len(rho) + 1,
    )

    rho2 = rho ** 2

    soma_fac2 = np.cumsum(
        rho2
    )

    soma_anterior = np.concatenate(
        ([0.0], soma_fac2[:-1])
    )

    sigma_rho = np.sqrt(
        (
            1.0
            + 2.0 * soma_anterior
        )
        / N
    )

    limite_zhang = (
        Z_CRIT * sigma_rho
    )

    abs_rho = np.abs(
        rho
    )

    significativo = (
        abs_rho
        > limite_zhang
    )

    indices_sig = np.where(
        significativo
    )[0]

    if len(indices_sig) == 0:

        n_c = 0
        n_t = 0
        soma_zhang = 0.0
        g = 1.0

    else:

        n_c = int(
            lags[
                indices_sig[-1]
            ]
        )

        n_t = min(
            n_c,
            max_lag,
        )

        mask = (
            lags <= n_t
        )

        termos = (
            (N - lags[mask])
            * rho[mask]
        )

        soma_zhang = float(
            np.sum(termos)
        )

        g = float(
            1.0
            + (2.0 / N)
            * soma_zhang
        )

    if g <= 0:

        neff = np.nan
        uA = np.nan

        alerta = (
            "g <= 0: revisar a janela."
        )

    else:

        neff = (
            N / g
        )

        uA = (
            ep_indep
            * math.sqrt(g)
        )

        alerta = ""

    tabela = pd.DataFrame({
        "Lag": lags,
        "FAC": rho,
        "FAC_quadrado": rho2,
        "Soma_FAC2": soma_fac2,
        "Soma_anterior": soma_anterior,
        "Sigma_rho": sigma_rho,
        "Limite_Zhang": limite_zhang,
        "Abs_FAC": abs_rho,
        "Significativo":
            significativo.astype(int),
        "Termo_Zhang":
            (N - lags) * rho,
        "Usado_ate_nt":
            (lags <= n_t).astype(int),
    })

    resumo = {
        "N": N,
        "Media": media,
        "DesvPad": s,
        "EP_indep": ep_indep,
        "max_lag_N4": max_lag,
        "n_c": n_c,
        "n_t": n_t,
        "Soma_Termo_Zhang":
            soma_zhang,
        "g": g,
        "N_eff": neff,
        "uA_Zhang": uA,
        "Razao_uA_EP":
            (
                uA / ep_indep
                if np.isfinite(uA)
                else np.nan
            ),
        "Alerta_Zhang": alerta,
    }

    return resumo, tabela


# ============================================================
# EXECUÇÃO COMPLETA
# ============================================================

def analisar_experimento():

    print("\n========================================")
    print(f"ANÁLISE COMPLETA — {NOME_EXPERIMENTO}")
    print("========================================\n")

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df = carregar_dados()

    # ETAPA 1 — autocorrelação do experimento completo
    print("\n2) Verificando autocorrelação do experimento completo...")
    autocorr_total, acf_total = verificar_autocorrelacao(
        df[COLUNA_FLUXO].to_numpy(),
        "Experimento_completo",
    )

    print(f"   FAC lag 1 = {autocorr_total['FAC_lag1']:.6f}")
    print(
        "   Autocorrelação detectada: "
        f"{autocorr_total['Autocorrelacionada']}"
    )

    # ETAPA 2 — pós-transiente definido manualmente
    print("\n3) Recortando o pós-transiente...")
    pos = separar_pos_transiente(df)

    print(
        f"   Início definido manualmente: "
        f"{INICIO_POS_TRANSIENTE_DIAS:.6f} dias"
    )
    print(
        f"   Tamanho da janela principal: "
        f"{TAMANHO_JANELA_HORAS:g} h"
    )
    print(
        f"   Tamanho da subjanela de validação: "
        f"{TAMANHO_SUBJANELA_HORAS:g} h"
    )

    autocorr_pos, acf_pos = verificar_autocorrelacao(
        pos[COLUNA_FLUXO].to_numpy(),
        "Pos_transiente",
    )

    # ETAPA 3 — todas as validações do tamanho da janela
    print("\n4) Validando o tamanho escolhido para as janelas...")

    stats_sub = estatisticas_subjanelas(pos)
    stats_jan = estatisticas_janelas(pos)

    comparacao_sub_jan, resumo_rep = validar_representatividade(
        stats_sub,
        stats_jan,
    )

    tabela_anova, resumo_anova = anova_posicao_subjanela(
        stats_sub
    )

    print(
        f"   ANOVA - p posição da subjanela = "
        f"{resumo_anova['p_Posicao_subjanela']:.6g}"
    )
    print(
        f"   Erro máximo subjanela x janela = "
        f"{resumo_rep['Erro_max_pct']:.4f}%"
    )
    print(
        f"   P95 do erro = "
        f"{resumo_rep['Erro_P95_pct']:.4f}%"
    )

    validacao_global = bool(
        resumo_anova["Sem_efeito_sistematico_posicao"]
        and resumo_rep["Representatividade_aprovada"]
    )

    print(
        "   Validação global do tamanho da janela: "
        f"{validacao_global}"
    )

    # ETAPA 4 + 5 — validação individual e Zhang
    print("\n5) Validando cada janela e aplicando Zhang...")

    resultados = []
    tabelas_acf_zhang = {}

    for janela, sub in pos.groupby("Janela", sort=True):
        janela = int(janela)
        print(f"   Janela {janela}...")

        stats_sub_janela = stats_sub.loc[
            stats_sub["Janela"] == janela
        ].copy()

        validacao = validar_janela(
            janela,
            sub,
            stats_sub_janela,
        )

        serie = sub[COLUNA_FLUXO].dropna().to_numpy(dtype=float)

        resumo_zhang, tabela_zhang = zhang_incerteza(serie)

        resultado = {
            "Experimento": NOME_EXPERIMENTO,
            "Janela": janela,
            **validacao,
            **resumo_zhang,
        }

        if not validacao["Valida_para_Zhang"]:
            resultado["Alerta_validacao"] = (
                "Janela não passou em todos os critérios prévios; "
                "uA_Zhang não deve ser reportada sem revisão."
            )
        else:
            resultado["Alerta_validacao"] = ""

        resultados.append(resultado)
        tabelas_acf_zhang[janela] = tabela_zhang

    resultados = pd.DataFrame(resultados)

    # SAÍDAS
    print("\n6) Salvando resultados...")

    nome_base = (
        f"{NOME_EXPERIMENTO}_"
        f"{TAMANHO_JANELA_HORAS:g}h"
    ).replace(".", "p")

    df.to_csv(
        PASTA_SAIDA / f"{nome_base}_dados_preparados.csv",
        sep=";",
        decimal=",",
        index=False,
    )

    pos.to_csv(
        PASTA_SAIDA / f"{nome_base}_pos_transiente.csv",
        sep=";",
        decimal=",",
        index=False,
    )

    resultados.to_csv(
        PASTA_SAIDA / f"{nome_base}_resultado_Zhang.csv",
        sep=";",
        decimal=",",
        index=False,
    )

    resumo_global = pd.DataFrame([{
        "Experimento": NOME_EXPERIMENTO,
        "Inicio_pos_transiente_dias": INICIO_POS_TRANSIENTE_DIAS,
        "Tamanho_janela_horas": TAMANHO_JANELA_HORAS,
        "Tamanho_subjanela_horas": TAMANHO_SUBJANELA_HORAS,
        **{
            f"Autocorr_total_{k}": v
            for k, v in autocorr_total.items()
        },
        **{
            f"Autocorr_pos_{k}": v
            for k, v in autocorr_pos.items()
        },
        **resumo_anova,
        **resumo_rep,
        "Validacao_global_janela": validacao_global,
    }])

    excel = PASTA_SAIDA / f"{nome_base}_analise_estatistica_completa.xlsx"

    with pd.ExcelWriter(excel, engine="openpyxl") as writer:

        resumo_global.to_excel(
            writer, sheet_name="Resumo_global", index=False
        )

        acf_total.to_excel(
            writer, sheet_name="ACF_total", index=False
        )

        acf_pos.to_excel(
            writer, sheet_name="ACF_pos", index=False
        )

        stats_sub.to_excel(
            writer, sheet_name="Estat_subjanelas", index=False
        )

        stats_jan.to_excel(
            writer, sheet_name="Estat_janelas", index=False
        )

        comparacao_sub_jan.to_excel(
            writer, sheet_name="Comparacao_sub_jan", index=False
        )

        tabela_anova.to_excel(
            writer, sheet_name="ANOVA_janelas", index=False
        )

        resultados.to_excel(
            writer, sheet_name="Resumo_Zhang", index=False
        )

        for janela, tabela in tabelas_acf_zhang.items():
            tabela.to_excel(
                writer,
                sheet_name=f"Zhang_J{janela}"[:31],
                index=False,
            )

    print("\nAnálise concluída.")
    print(f"\nArquivo principal:\n{excel}")

    colunas = [
        "Janela",
        "Inicio_janela_dias",
        "Fim_janela_dias",
        "Primeiro_dado_dias",
        "Ultimo_dado_dias",
        "ADF",
        "ADF_p",
        "ADF_estacionaria",
        "Janela_completa",
        "Todas_subjanelas_presentes",
        "Erro_max_sub_vs_janela_pct",
        "Valida_para_Zhang",
        "N",
        "Media",
        "DesvPad",
        "n_c",
        "n_t",
        "g",
        "N_eff",
        "uA_Zhang",
        "Alerta_Zhang",
        "Alerta_validacao",
    ]

    print("\nResumo por janela:")
    print(resultados[colunas].to_string(index=False))


if __name__ == "__main__":
    analisar_experimento()
