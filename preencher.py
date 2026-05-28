import pandas as pd

arquivo_entrada = (
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\measurements_processado2.csv"
)

arquivo_saida = (
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\measurements_processado2_preenchido.csv"
)

df = pd.read_csv(
    arquivo_entrada,
    sep="/"
)

df.columns = df.columns.str.strip()

print("Colunas encontradas:")
print(df.columns.tolist())

# Garante que tempo_segundos seja numérico
df["tempo_segundos"] = pd.to_numeric(df["tempo_segundos"], errors="coerce")

# Cria coluna Tempo_dias
df["Tempo_dias"] = df["tempo_segundos"] / 86400

# Garante que as colunas de interesse sejam numéricas
colunas_processar = [
    "fluxo_permeado",
    "condhot",
    "GOR_ant",
    "SEC_termico_kWh_m3"
]

for coluna in colunas_processar:
    if coluna in df.columns:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    else:
        print(f"Atenção: coluna não encontrada: {coluna}")

origem_inicio = 4
origem_fim = 6

destino_inicio = 2
destino_fim = 4

trecho_origem = df[
    (df["Tempo_dias"] >= origem_inicio) &
    (df["Tempo_dias"] <= origem_fim)
].copy()

# Desloca o trecho 4–6 dias para 2–4 dias
deslocamento = origem_inicio - destino_inicio

trecho_origem["Tempo_dias"] = trecho_origem["Tempo_dias"] - deslocamento
trecho_origem["tempo_segundos"] = trecho_origem["Tempo_dias"] * 86400

# Remove os dados originais entre 2 e 4 dias
df_sem_destino = df[
    ~(
        (df["Tempo_dias"] >= destino_inicio) &
        (df["Tempo_dias"] <= destino_fim)
    )
]

# Junta os dados originais com o trecho deslocado
df_final = pd.concat(
    [df_sem_destino, trecho_origem],
    ignore_index=True
)

df_final = df_final.sort_values(by="Tempo_dias")

df_final.to_csv(
    arquivo_saida,
    sep="/",
    index=False
)

print("\nArquivo gerado com sucesso.")
print(f"Local do arquivo:\n{arquivo_saida}")