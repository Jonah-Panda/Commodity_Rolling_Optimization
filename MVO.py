import pandas as pd
import os
import numpy as np
from math import sqrt

cwd = os.getcwd()
# com_code = "SI"

def single_day_roll(dte_roll, com_code):
    # gets list of all futures contracts for underlying commodity
    contract_list = os.listdir("{}\Data\{}".format(cwd, com_code))

    # getting initial future contract df
    df = pd.read_csv("{}\Data\{}\{}".format(cwd, com_code, contract_list[0]), index_col=0)
    try:
        start_date = df.iloc[90, 0]
    except:
        start_date = df.iloc[-5, 0]
    df = df[df['Date'] > start_date]

    # itterating through all contracts to create continuous timeframe.
    for i in range(1, len(contract_list)):

        df2 = pd.read_csv("{}\Data\{}\{}".format(cwd, com_code, contract_list[i]), index_col=0)
        df2['Close'] = df2['Close'].fillna(0)
        if df2.iloc[0, 1] == 0:
            df2 = df2[1:]

        df = df.merge(df2, on='Date', how='left')
        df['Close_y'] = df['Close_y'].fillna(0)
        df['Volume_y'] = df['Volume_y'].fillna(0)

        roll_date = df.iloc[dte_roll, 0]
        new_contract_weight = df.loc[dte_roll, 'Close_x'] / df.loc[dte_roll, 'Close_y']
        # print(new_contract_weight)
        df['Weight1'] = np.where(df['Date'] < roll_date, 1, 0)
        df['Weight2'] = np.where(df['Date'] >= roll_date, new_contract_weight, 0)
        df['Close'] = df['Close_x'] * df['Weight1'] + df['Close_y'] * df['Weight2']
        df['Volume'] = round(df['Volume_x'] * df['Weight1'] + df['Volume_y'] * df['Weight2'], 0)
        # print(df.iloc[dte_roll]) # checks that the contract roll is the same by dollars
        df = df[['Date', 'Close', 'Volume']]

        df2 = df2[df2['Date'] > df['Date'].max()]
        df2['Close'] = df2['Close'] * new_contract_weight
        df = pd.concat([df2, df])
        df.reset_index(inplace=True, drop=True)

    return df

def multi_day_roll(n_days, last_dte_roll, com_code):
    df = single_day_roll(last_dte_roll, com_code)

    for i in range(last_dte_roll+1, last_dte_roll+n_days):
        df2 = single_day_roll(i, com_code)
        df2 = df2[['Date', 'Close']]

        # Merging different individual day roll
        df = df.merge(df2, on='Date', how='left')

        # Adding close to cumulative sum that will be divided after cumulative sum
        df['Close'] = df['Close_x'] + df['Close_y']
        df = df[['Date', 'Close', 'Volume']]

    # Dividing cumulaitve sum by number of days to roll over
    df['Close'] = df['Close'] / n_days

    return df

def get_mean_variance(df):
    df['Close_minus_1'] = df['Close'].shift(-1)
    df['Return'] = (df['Close'] - df['Close_minus_1']) / df['Close_minus_1']

    mean = df.loc[0:len(df)-2, 'Return'].mean()
    var = df.loc[0:len(df)-2, 'Return'].var()
    meanVariance = mean/(sqrt(252) * var)
    return meanVariance

def get_commodity_codes_df():
    file_name = "{}\FNCE 449 - Final Project.xlsx".format(cwd)
    sheet_name = "Building Table"
    df = pd.read_excel(io=file_name, sheet_name=sheet_name, header=None, skiprows=27)
    df = df[[0, 1]]
    df.columns = ["Commodity", "Code"]
    df = df[0:27]

    # Dropping rows without enough liquidity for roll calculations.
    a = ["CC", "LB", "LS", "JO", "XB", "SM"] 
    df = df[~df['Code'].isin(a)]
    df.reset_index(inplace=True, drop=True)

    return df


# df = single_day_roll(3, com_code)
# print(df)

# df = multi_day_roll(3, 3, com_code)
# print(df)

com_code = "SI"
df = multi_day_roll(15, 20, com_code)
print(df)
df = multi_day_roll(14, 21, com_code)
print(df)
exit()




rows = []
start_roll_date = []
end_roll_date = []
n_days = []
for i in range(1, 35):
    for j in range(1, 35):
        if i < j:
            rows = rows + ["{}-{}".format(i, j)]
            start_roll_date = start_roll_date + [i]
            end_roll_date = end_roll_date + [j]
            n_days = n_days + [j-i+1]
    if i > 1:
        break

Com_codes = get_commodity_codes_df()
Com_codes = Com_codes['Code'].tolist()


# df = pd.DataFrame(columns=Com_codes, index=rows)
# print(df)

df = pd.read_csv('Multi_day_MV.csv', index_col=0)

for code in Com_codes:
    print()
    for id in rows:
        index = rows.index(id)

        if index % 5 == 0:
            df.to_csv('{}\Multi_day_MV.csv'.format(cwd))
            print(df)

        print('{} - {}%'.format(code, round(100*index/len(rows), 2)), end='\r')

        try:
            temp_df = multi_day_roll(start_roll_date[index], end_roll_date[index], code)
            meanVar = get_mean_variance(temp_df)
        except:
            meanVar = 0
        df.loc[id, "{}".format(code)] = meanVar

#df.to_csv('{}\Multi_day_MV2.csv'.format(cwd))
print(df)

exit()
