import pandas as pd
import os
import numpy as np

cwd = os.getcwd()
com_code = "CL"

def single_day_roll(dte_roll, com_code):

    contract_list = os.listdir("{}\Data\{}".format(cwd, com_code))
    # contract_list = contract_list[0:5]

    # getting initial future contract df
    df = pd.read_csv("{}\Data\{}\{}".format(cwd, com_code, contract_list[0]), index_col=0)
    start_date = df.iloc[90, 0]

    # itterating through all contracts to create continuous timeframe.
    for i in range(1, len(contract_list)):

        df2 = pd.read_csv("{}\Data\{}\{}".format(cwd, com_code, contract_list[i]), index_col=0)
        df2['Close'] = df2['Close'].fillna(0)
        if df2.iloc[0, 1] == 0:
            df2 = df2[1:]

        df = df.merge(df2, on='Date', how='left')
        df = df[df['Date'] > start_date]
        df['Close_y'] = df['Close_y'].fillna(0)
        df['Volume_y'] = df['Volume_y'].fillna(0)

        roll_date = df.iloc[dte_roll, 0]
        df['Weight1'] = np.where(df['Date'] < roll_date, 1, 0)
        df['Weight2'] = np.where(df['Date'] >= roll_date, 1, 0)
        df['Close'] = df['Close_x'] * df['Weight1'] + df['Close_y'] * df['Weight2']
        df['Volume'] = round(df['Volume_x'] * df['Weight1'] + df['Volume_y'] * df['Weight2'], 0)
        df = df[['Date', 'Close', 'Volume']]

        df2 = df2[df2['Date'] > df['Date'].max()]
        df = pd.concat([df2, df])
        df.reset_index(inplace=True, drop=True)

    return df


df = single_day_roll(3, com_code)
print(df)