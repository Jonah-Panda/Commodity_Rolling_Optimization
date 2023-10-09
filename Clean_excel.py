import pandas as pd
import os
import datetime
from numpy import nan

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
def month_zero(x):
    str_month =  str(x.month)
    return str_month if len(str_month) == 2 else '0{}'.format(str_month)
def day_zero(x):
    str_day = str(x.day)
    return str_day if len(str_day) == 2 else '0{}'.format(str_day)


# df_comdty = get_commodity_codes_df()
# create_folders_for_comdty_data(df_comdty)

# print(df_comdty)

def open_sheet(sheet_name):
    # Opening Sheet
    file_name = "{}\FNCE 449 - Final Project.xlsx".format(cwd)
    df = pd.read_excel(io=file_name, sheet_name=sheet_name)

    # Removing first column
    df = df.iloc[:, 1:]

    return df

df = open_sheet("CT")

def clean_contract2(df, sheet_name):
    # Removing rows with no data
    df["na_count"] = df.isna().sum(axis=1)
    df = df[df.na_count != 3]
    df = df.drop(["na_count"], axis=1)
    df['Volume'] = df['Volume'].replace({nan : 0})

    df.reset_index(inplace=True, drop=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Finding Last Tradable Day for all Securities
    expiry_date = df.loc[0, "Date"]
    string_expiry_date = "{}-{}-{}".format(expiry_date.year, month_zero(expiry_date), day_zero(expiry_date))
    # print(string_expiry_date)
    if expiry_date <= datetime.datetime(2023, 10, 5):
        df.to_csv("{}\Data\{}\{}.csv".format(cwd, sheet_name, string_expiry_date))
    return df

def export_contracts(df, sheet_name):
    col_names = df.columns.tolist()
    for i in range(0, len(col_names), 3):
        contract_df = df[col_names[i:i+3]]

        # contract_name = contract_df.iloc[0, 0]
        # data_end_date = contract_df.iloc[2, 0]
        # Removing Header from Bloomberg Data Pulling
        contract_df = contract_df.iloc[5:, :]

        # Renaming Columns for contract
        contract_df.columns = ['Date', "Close", "Volume"]
        contract_df = contract_df.iloc[1:]

        if contract_df.iloc[0, 0] == "#N/A Invalid Security":
            # print(contract_df.head(2))
            continue
        elif contract_df["Volume"].isna().sum() == len(contract_df):
            # print(contract_df.head(2))
            continue
        else:
            # print(contract_df)
            clean_contract2(contract_df, sheet_name)

    return

export_contracts(df, "CT")
