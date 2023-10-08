import pandas as pd
import os

cwd = os.getcwd()

def get_commodity_codes_df():
    file_name = "{}\FNCE 449 - Final Project.xlsx".format(cwd)
    sheet_name = "Building Table"
    df = pd.read_excel(io=file_name, sheet_name=sheet_name, header=None, skiprows=27)
    df = df[[0, 1]]
    df.columns = ["Commodity", "Code"]
    df = df[0:27]
    return df

def create_folders_for_comdty_data(df_comdty):
    for index, row in df_comdty.iterrows():
        comdty_code = row['Code']
        if comdty_code not in os.listdir('{}\Data'.format(cwd)):
            os.mkdir("{}\Data\{}".format(cwd, comdty_code))
    return


df_comdty = get_commodity_codes_df()
create_folders_for_comdty_data(df_comdty)

print(df_comdty)

