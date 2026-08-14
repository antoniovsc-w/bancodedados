import pandas as pd

arquivo_entrada = (
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M1\dados_M1.csv"
)

arquivo_saida = (
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M1\dados_final_M1_preenchido.csv"
)

# ============================================================
# 1. LEITURA
# ============================================================

df = pd.read_csv(
    arquivo_entrada,
    sep="/",
    dtype=str
)

df.columns = df.columns.str.strip()

print("Colunas encontradas:")
print(df.columns.tolist())


# ============================================================
# 2. FUNÇÃO PARA CONVERTER NÚMEROS
# ============================================================

def converter_numerico(serie):

    return pd.to_numeric(
        serie
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


# ============================================================
# 3. CONVERTE TEMPO_DIAS
# ============================================================

df["Tempo_dias"] = converter_numerico(
    df["Tempo_dias"]
)


# ============================================================
# 4. CONVERTE AS COLUNAS NUMÉRICAS
# ============================================================

colunas_processar = [
    "fluxo_permeado",
    "condhot",
    "GOR_ant",
    "SEC_termico_kWh_m3"
]

for coluna in colunas_processar:

    if coluna in df.columns:

        print(f"\nProcessando coluna: {coluna}")

        print("Primeiros valores ANTES da conversão:")
        print(df[coluna].head())

        df[coluna] = converter_numerico(
            df[coluna]
        )

        print("Primeiros valores DEPOIS da conversão:")
        print(df[coluna].head())

        print(
            f"Valores válidos: {df[coluna].notna().sum()} "
            f"de {len(df)}"
        )

    else:

        print(
            f"Atenção: coluna não encontrada: {coluna}"
        )


# ============================================================
# 5. INTERVALOS
# ============================================================

origem_inicio = 2.60
origem_fim = 3.21

destino_inicio = 1.20
destino_fim = 1.81


# ============================================================
# 6. SELECIONA TRECHO DE ORIGEM
# ============================================================

trecho_origem = df[
    (df["Tempo_dias"] >= origem_inicio) &
    (df["Tempo_dias"] <= origem_fim)
].copy()

print(
    f"\nQuantidade de linhas no trecho de origem: "
    f"{len(trecho_origem)}"
)


# ============================================================
# 7. DESLOCA O TEMPO
# ============================================================

deslocamento = origem_inicio - destino_inicio

trecho_origem["Tempo_dias"] = (
    trecho_origem["Tempo_dias"]
    - deslocamento
)


# Atualiza tempo_segundos apenas se a coluna existir
if "tempo_segundos" in trecho_origem.columns:

    trecho_origem["tempo_segundos"] = (
        trecho_origem["Tempo_dias"] * 86400
    )


# ============================================================
# 8. REMOVE INTERVALO QUE SERÁ SUBSTITUÍDO
# ============================================================

df_sem_destino = df[
    ~(
        (df["Tempo_dias"] >= destino_inicio) &
        (df["Tempo_dias"] <= destino_fim)
    )
].copy()


# ============================================================
# 9. JUNTA OS DADOS
# ============================================================

df_final = pd.concat(
    [
        df_sem_destino,
        trecho_origem
    ],
    ignore_index=True
)


# ============================================================
# 10. ORDENA
# ============================================================

df_final = (
    df_final
    .sort_values("Tempo_dias")
    .reset_index(drop=True)
)


# ============================================================
# 11. VERIFICA FLUXO ANTES DE SALVAR
# ============================================================

print("\nFluxo permeado no arquivo final:")

print(
    df_final[
        [
            "Tempo_dias",
            "fluxo_permeado"
        ]
    ].head(20)
)

print(
    "\nQuantidade de valores válidos de fluxo_permeado:",
    df_final["fluxo_permeado"].notna().sum()
)


# ============================================================
# 12. SALVA
# ============================================================

df_final.to_csv(
    arquivo_saida,
    sep="/",
    index=False,
    decimal=","
)

print("\nArquivo gerado com sucesso.")
print(f"Local:\n{arquivo_saida}")