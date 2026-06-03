import pandas as pd

df = pd.read_csv(
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\measurements_processado4_preenchido.csv",
    sep="/"
)

df.to_csv(
    r"C:\Users\Administrador\Desktop\EXPERIMENTOSLONGOPRAZO\M3\measurements_processado4_minitab.csv",
    sep="/",
    decimal=",",
    index=False
)