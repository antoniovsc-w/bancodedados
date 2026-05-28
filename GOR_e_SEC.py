import pandas as pd
import numpy as np

ARQUIVO_CSV = r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\measurements_processado2.csv"

SALINIDADE = 47.0          # g/kg
PRESSAO = 0.101325         # MPa

VAZAO_QUENTE_L_MIN = 2.5   # alterar para a vazão real
VAZAO_FRIO_L_MIN = 2.5   # alterar para a vazão real
AREA_MEMBRANA_M2 = 0.058   # área da membrana

# Colunas
T_H_IN = "temp_1"
T_H_OUT = "temp_2"
T_C_IN = "temp_3"
T_C_OUT = "temp_4"
FLUXO_PERMEADO = "fluxo_permeado"


# ==========================================================
# DENSIDADE DA SALMOURA
# Baseado em Nayar et al. (2016)
# ==========================================================
def densidade_salmoura(T, S=SALINIDADE, P=PRESSAO):
    """
    Densidade da salmoura em kg/m³.
    T em °C
    S em g/kg
    P em MPa
    """

    T = np.asarray(T, dtype=float)

    P0 = 0.101325
    S_kgkg = S / 1000

    a1 = 9.999e2
    a2 = 2.034e-2
    a3 = -6.162e-3
    a4 = 2.261e-5
    a5 = -4.657e-8

    b1 = 8.020e2
    b2 = -2.001
    b3 = 1.677e-2
    b4 = -3.060e-5
    b5 = -1.613e-5

    rho_p0 = (
        a1
        + a2*T
        + a3*T**2
        + a4*T**3
        + a5*T**4
        + b1*S_kgkg
        + b2*S_kgkg*T
        + b3*S_kgkg*T**2
        + b4*S_kgkg*T**3
        + b5*(S_kgkg**2)*(T**2)
    )

    c1 = 5.0792e-4
    c2 = -3.4168e-6
    c3 = 5.6931e-8
    c4 = -3.7263e-10
    c5 = 1.4465e-12
    c6 = -1.7058e-15

    c7 = -1.3389e-6
    c8 = 4.8603e-9
    c9 = -6.8039e-13

    d1 = -1.1077e-6
    d2 = 5.5584e-9
    d3 = -4.2539e-11
    d4 = 8.3702e-9

    Fp = np.exp(
        (P - P0) * (
            c1
            + c2*T
            + c3*T**2
            + c4*T**3
            + c5*T**4
            + c6*T**5
            + S * (c7 + c8*T + c9*T**2)
        )
        + ((P**2 - P0**2) / 2) * (
            d1
            + d2*T
            + d3*T**2
            + d4*S
        )
    )

    return rho_p0 * Fp


# ==========================================================
# CALOR ESPECÍFICO DA SALMOURA
# ==========================================================
def cp_salmoura(T, S=SALINIDADE, P=PRESSAO):
    """
    Calor específico da salmoura em J/(kg.K)
    """

    T = np.asarray(T, dtype=float)

    P0 = 0.101325

    a1 = -3.1118
    a2 = 0.0157
    a3 = 5.1014e-5
    a4 = -1.0302e-6

    a5 = 0.0107
    a6 = -3.9716e-5
    a7 = 3.2088e-8
    a8 = 1.0119e-9

    A = 5328 - 0.976*S + 4.04e-3*S**2
    B = -6.913 + 7.351e-3*S - 3.15e-5*S**2
    C = 9.6e-3 - 1.927e-5*S + 8.23e-6*S**2
    D = 2.5e-6 + 1.666e-8*S - 7.125e-10*S**2

    Tk = T + 273.15

    cp_p0 = (
        A
        + B*Tk
        + C*Tk**2
        + D*Tk**3
    )

    cp = cp_p0 + (P - P0) * (
        a1
        + a2*T
        + a3*T**2
        + a4*T**3
        + S * (
            a5
            + a6*T
            + a7*T**2
            + a8*T**3
        )
    )

    return cp


# ==========================================================
# ENTALPIA DE EVAPORAÇÃO
# ==========================================================
def h_evaporacao_agua(T):
    """
    Entalpia de evaporação da água [J/kg]
    Aproximação:
    h_fg = 2501 - 2.361*T  [kJ/kg]
    """

    return (2501 - 2.361*T) * 1000


# ==========================================================
# LEITURA DO CSV
# ==========================================================
df = pd.read_csv(
    ARQUIVO_CSV,
    sep="/",
    engine="python"
)

colunas = [
    T_H_IN,
    T_H_OUT,
    T_C_IN,
    T_C_OUT,
    FLUXO_PERMEADO
]

for col in colunas:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(subset=colunas)


# ==========================================================
# TEMPERATURA MÉDIA DO LADO QUENTE
# ==========================================================
df["T_media_quente_C"] = (
    df[T_H_IN]
    + df[T_H_OUT]
) / 2


# ==========================================================
# PROPRIEDADES DA SALMOURA
# ==========================================================
df["rho_salmoura_kg_m3"] = densidade_salmoura(
    df["T_media_quente_C"]
)

df["cp_salmoura_J_kgK"] = cp_salmoura(
    df["T_media_quente_C"]
)


# ==========================================================
# VAZÃO MÁSSICA DO LADO QUENTE
# ==========================================================
vazao_quente_m3_s = (
    VAZAO_QUENTE_L_MIN
    / 1000
    / 60
)

df["m_dot_quente_kg_s"] = (
    df["rho_salmoura_kg_m3"]
    * vazao_quente_m3_s
)



# ==========================================================
# CALOR SENSÍVEL DO LADO QUENTE
# ==========================================================
df["deltaT_quente_C"] = (
    df[T_H_IN]
    - df[T_H_OUT]
)

df["Q_quente_W"] = (
    df["m_dot_quente_kg_s"]
    * df["cp_salmoura_J_kgK"]
    * df["deltaT_quente_C"]
)


# ==========================================================
# TEMPERATURA DE REFERÊNCIA
# ==========================================================
df["T_ref_C"] = (
    (
        (df[T_H_IN] + df[T_H_OUT]) / 2
    )
    +
    (
        (df[T_C_IN] + df[T_C_OUT]) / 2
    )
) / 2


# ==========================================================
# ENTALPIA DE EVAPORAÇÃO
# ==========================================================
df["h_evap_J_kg"] = h_evaporacao_agua(
    df["T_ref_C"]
)


# ==========================================================
# FLUXO DE PERMEADO
# Assume fluxo_permeado em L/(m².h)
# ==========================================================
df["vazao_permeado_m3_s"] = (
    df[FLUXO_PERMEADO]
    * AREA_MEMBRANA_M2
    / 1000
    / 3600
)


# ==========================================================
# MASSA ESPECÍFICA DA ÁGUA
# ==========================================================
df["rho_agua_kg_m3"] = 997.0


# ==========================================================
# CALOR ESPECÍFICO DA ÁGUA
# ==========================================================
df["cp_agua_J_kgK"] = 4180

# ==========================================================
# VAZÃO MÁSSICA DO LADO FRIO
# ==========================================================
vazao_frio_m3_s = (
    VAZAO_FRIO_L_MIN
    / 1000
    / 60
)

df["m_dot_frio_kg_s"] = (
    df["rho_agua_kg_m3"]
    * vazao_frio_m3_s
)

# ==========================================================
# CALOR SENSÍVEL DO LADO FRIO
# ==========================================================
df["deltaT_frio_C"] = (
    df[T_C_IN]
    - df[T_C_OUT]
)

df["Q_frio_W"] = (
    df["m_dot_frio_kg_s"]
    * df["cp_agua_J_kgK"]
    * df["deltaT_frio_C"]
)


# ==========================================================
# VAZÃO MÁSSICA DE PERMEADO
# ==========================================================
df["m_dot_permeado_kg_s"] = (
    df["vazao_permeado_m3_s"]
    * df["rho_agua_kg_m3"]
)


# ==========================================================
# CALOR DE EVAPORAÇÃO
# ==========================================================
df["Q_evaporacao_W"] = (
    df["m_dot_permeado_kg_s"]
    * df["h_evap_J_kg"]
)

# ==========================================================
# GOR - Gained Output Ratio
# ==========================================================
df["GOR_ant"] = np.where(
    df["Q_quente_W"] > 0,
    df["Q_evaporacao_W"] / df["Q_quente_W"],
    np.nan
)


# ==========================================================
# SEC TÉRMICO
# Unidade: kWh/m³
# SEC = energia térmica consumida / volume de permeado produzido
# ==========================================================
df["SEC_termico_kWh_m31"] = np.where(
    df["vazao_permeado_m3_s"] > 0,
    df["Q_quente_W"] / df["vazao_permeado_m3_s"] / 3.6e6,
    np.nan
)

df["SEC_termico_kWh_m3"] = (
    df["SEC_termico_kWh_m31"] / 1000
)

# ==========================================================
# BALANÇO Q
# ==========================================================
df["Balanco_Q"] = (
    df["Q_quente_W"]
    + df["Q_frio_W"]
    - df["Q_evaporacao_W"]
)


# ==========================================================
# EXPORTAÇÃO
# Sobrescreve o CSV original adicionando as novas colunas
# ==========================================================
df.to_csv(
    ARQUIVO_CSV,
    index=False,
    sep="/"
)


# ==========================================================
# RESULTADOS
# ==========================================================
print(df[[
    T_H_IN,
    T_H_OUT,
    T_C_IN,
    T_C_OUT,
    "rho_salmoura_kg_m3",
    "cp_salmoura_J_kgK",
    "Q_quente_W",
    "Q_evaporacao_W",
    "GOR_ant",
    "SEC_termico_kWh_m3",
    "Balanco_Q"
]].head())

print("\nArquivo atualizado")