import pandas as pd
import os

cwd = os.getcwd()

file_name = "{}\FNCE 449 - Final Project.xlsx".format(cwd)
sheet_name = "Building Table"
df = pd.read_excel(io=file_name, sheet_name=sheet_name, header=None, skiprows=27)
df = df[[0, 1]]
df.columns = ["Commodity", "Code"]
df = df[0:28]
df_comdty = df
print(df_comdty)