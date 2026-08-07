import pandas as pd

arquivo_entrada = r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M2\Informações\Resultado por segundo final - antonio.csv"
arquivo_saida = r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M2\Informações\Resultado_por_segundo_convertido.csv"

# Leitura do CSV com separador //
df = pd.read_csv(
    arquivo_entrada,
    sep="//",
    engine="python"
)

# Remove espaços e aspas dos nomes das colunas
df.columns = (
    df.columns
    .str.strip()
    .str.replace('"', '', regex=False)
)

# Salva usando / como separador
df.to_csv(
    arquivo_saida,
    sep="/",
    index=False
)

print("Conversão concluída.")
print("Arquivo salvo em:")
print(arquivo_saida)

print("\nColunas encontradas:")
print(df.columns.tolist())