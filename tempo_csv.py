import pandas as pd

# Arquivo exportado pelo DBeaver
arquivo_csv = r"C:\Users\Administrador\Downloads\database11teste.csv"

# Arquivo de saída
saida_csv = r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\measurements_processado.csv"

# Ler CSV exportado pelo DBeaver
df = pd.read_csv(
    arquivo_csv,
    sep="//",
    engine="python",
    encoding="utf-8-sig"
)

# Corrigir nomes das colunas e remover aspas
df.columns = df.columns.str.strip().str.replace('"', '', regex=False)

# Remover aspas dos valores
df = df.replace('"', '', regex=True)

print("Colunas corrigidas:")
print(df.columns.tolist())

# Converter timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Converter colunas numéricas
colunas_numericas = [
    "temp_1", "temp_2", "temp_3", "temp_4",
    "temp_5", "temp_6", "temp_7", "temp_8",
    "pressao_1", "pressao_2", "pressao_3", "pressao_4", "pressao_5",
    "condhot", "sec", "gor", "fluxo_permeado"
]

for coluna in colunas_numericas:
    if coluna in df.columns:
        df[coluna] = (
            df[coluna]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

# Ordenar por timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# Criar colunas anteriores
df["timestamp_anterior"] = df["timestamp"].shift(1)
df["temp_1_anterior"] = df["temp_1"].shift(1)

# Preencher condhot com o último valor válido
df["condhot"] = df["condhot"].ffill()

# Condição para calcular delta_segundos
condicao_delta = (
    df["timestamp_anterior"].notna()
    & (df["temp_1"] >= 60)
    & (df["temp_1_anterior"] >= 60)
    & (df["timestamp"] > df["timestamp_anterior"])
)

# Calcular delta_segundos
df["delta_segundos"] = pd.NA

df.loc[condicao_delta, "delta_segundos"] = (
    df.loc[condicao_delta, "timestamp"]
    - df.loc[condicao_delta, "timestamp_anterior"]
).dt.total_seconds()

df["delta_segundos"] = pd.to_numeric(df["delta_segundos"], errors="coerce")

# Calcular tempo acumulado
df["tempo_segundos"] = df["delta_segundos"].cumsum()

# Aplicar filtros finais
measurements_processado = df[
    df["delta_segundos"].notna()
    & (df["fluxo_permeado"] > 0)
    & (df["fluxo_permeado"] < 40)
].copy()

# Colunas finais
colunas_finais = [
    "id",
    "timestamp",
    "temp_1",
    "temp_2",
    "temp_3",
    "temp_4",
    "temp_5",
    "temp_6",
    "temp_7",
    "temp_8",
    "pressao_1",
    "pressao_2",
    "pressao_3",
    "pressao_4",
    "pressao_5",
    "condhot",
    "sec",
    "gor",
    "fluxo_permeado",
    "timestamp_anterior",
    "temp_1_anterior",
    "delta_segundos",
    "tempo_segundos"
]

# Manter apenas colunas que existem no arquivo
colunas_existentes = [col for col in colunas_finais if col in measurements_processado.columns]

measurements_processado = measurements_processado[colunas_existentes]

# Salvar resultado
measurements_processado.to_csv(
    saida_csv,
    index=False,
    sep="/",
    encoding="utf-8-sig"
)

print("\nPrimeiras 50 linhas processadas:")
print(measurements_processado.head(50))

print(f"\nArquivo salvo em:\n{saida_csv}")