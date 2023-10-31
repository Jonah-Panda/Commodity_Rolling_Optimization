import pandas as pd
import os
import numpy as np
from numpy import where
from math import sqrt
import datetime

cwd = os.getcwd()
# com_code = "SI"
START_DATE = datetime.datetime(2013, 7, 1)
END_DATE = datetime.datetime(2020, 9, 1)

def month_zero(x):
    str_month =  str(x.month)
    return str_month if len(str_month) == 2 else '0{}'.format(str_month)
def day_zero(x):
    str_day = str(x.day)
    return str_day if len(str_day) == 2 else '0{}'.format(str_day)
def dt_to_string(date):
    return "{}-{}-{}".format(date.year, month_zero(date), day_zero(date))
def merge_futures_contracts(com_code):
    datelist = pd.bdate_range(start=START_DATE, end=END_DATE)
    df = pd.DataFrame(datelist, columns=['Date'])
    del datelist

    contract_list = os.listdir("{}\Data\{}".format(cwd, com_code))
    for file in contract_list:
        df_data = pd.read_csv("{}\Data\{}\{}".format(cwd, com_code, file), index_col=0)
        df_data['Date'] = pd.to_datetime(df_data['Date'])
        last_tradeable_day = df_data.loc[0, ['Date']][0]
        if last_tradeable_day > END_DATE:
            continue
        else:
            df_data.drop(columns=["Volume"], inplace=True)
            df_data.rename(columns={'Close':'{}'.format(dt_to_string(last_tradeable_day))}, inplace=True)
            df = df.merge(df_data, on='Date', how='left')

    # Dropping rows where markets are closed
    df['Sum'] = df.count(axis=1)
    drop_rows_NaN_count = min(df.Sum)
    df.drop(df[df.Sum == drop_rows_NaN_count].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.drop(columns=['Sum'], inplace=True)
    return df
def single_day_roll(dte_roll, df_com):
    col_names = df_com.columns.tolist()[1:]
    df = pd.DataFrame(columns=['Date', 'Close'])
    df.loc[0, "Date"] = df_com['Date'][0]
    df.loc[0, "Close"] = df_com['{}'.format(col_names[0])][0]

    # dte_roll = 5
    old_contract_weight = 1
    for i in range(len(col_names)-1):
        col = col_names[i]
        ltd = datetime.datetime(int(col[0:4]), int(col[5:7]), int(col[8:10]))
        df2 = df_com.loc[df_com["Date"] <= ltd, ["Date", "{}".format(col), "{}".format(col_names[i+1])]]
        df2.interpolate(method='pad', limit=10, inplace=True) # Padding for days with no trading volume

        roll_index = len(df2) - dte_roll
        roll_date = df2.loc[roll_index, ["Date"]][0]

        # HEY, IF YOU HAVE TIME YOU SHOULD TOTALLY COME BACK AND ADD VOLUME TRIGGERS AND CASH TRIGGERS TO THE MATCHING ALGO SO IT IS ACTUALLY REPRESENTATIVE
        close2 = df2.loc[df2['Date'] == roll_date, '{}'.format(col_names[i+1])]
        if np.isnan(close2).bool():
            return 0
        new_contract_weight = df2.loc[df2['Date'] == roll_date, '{}'.format(col)] / close2
        df2['Weight1'] = np.where(df2['Date'] < roll_date, old_contract_weight, 0)
        df2['Weight2'] = np.where(df2['Date'] >= roll_date, new_contract_weight, 0)
        old_contract_weight = new_contract_weight

        df2['Close'] = df2['{}'.format(col)] * df2['Weight1'] + df2['{}'.format(col_names[i+1])] * df2['Weight2']
        # print(df2)
        df2 = df2[['Date', 'Close']]

        df2 = df2[df2['Date'] > df['Date'].max()]
        df = pd.concat([df, df2])
    df.dropna(inplace=True)
    return df
# def multi_day_roll(n_days, last_dte_roll, df_com):
#     df_multi = single_day_roll(last_dte_roll, df_com)
#     if type(df_multi) == int:
#         return 0

#     for i in range(last_dte_roll+1, last_dte_roll+n_days):
#         df2_multi = single_day_roll(i, df_com)
#         if type(df2_multi) == int:
#             return 0

#         # Merging different individual day roll
#         df_multi = df_multi.merge(df2_multi, on='Date', how='left')

#         # Adding close to cumulative sum that will be divided after cumulative sum
#         df_multi['Close'] = df_multi['Close_x'] + df_multi['Close_y']
#         df_multi = df_multi[['Date', 'Close']]

#     # Dividing cumulaitve sum by number of days to roll over
#     df_multi['Close'] = df_multi['Close'] / n_days

#     return df_multi    

def get_mean_variance(df):
    if type(df) == int:
        return 0
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
def single_day_builder(df_com):
    df = single_day_roll(1, df_com)
    df.rename(columns={'Close':'{}'.format(1)}, inplace=True)
    df = df[["Date", '{}'.format(1)]]

    for i in range(2, 25+25+1):
        temp_df = single_day_roll(i, df_com)
        if type(temp_df) == int:
            df['{}'.format(i)] = 0
        else:
            temp_df.rename(columns={'Close':'{}'.format(i)}, inplace=True)
            temp_df = temp_df[["Date", '{}'.format(i)]]
            df = df.merge(temp_df, on='Date', how='left')

    return df
def multi_day_efficient(n_days, last_dte_roll, single_df):
    df_multi = single_df[["Date", "{}".format(last_dte_roll)]]
    if df_multi['{}'.format(last_dte_roll)].sum() == 0:
        return 0
    df_multi = df_multi.rename(columns={'{}'.format(last_dte_roll):'Close'})

    for i in range(last_dte_roll+1, last_dte_roll+n_days):
        df2_multi = single_df[["Date", "{}".format(i)]]
        if df2_multi['{}'.format(i)].sum() == 0:
            return 0

        df2_multi = df2_multi.rename(columns={'{}'.format(i):'Close'})
        # Merging different individual day roll
        df_multi = df_multi.merge(df2_multi, on='Date', how='left')

        # Adding close to cumulative sum that will be divided after cumulative sum
        df_multi['Close'] = df_multi['Close_x'] + df_multi['Close_y']
        df_multi = df_multi[['Date', 'Close']]

    # Dividing cumulaitve sum by number of days to roll over
    df_multi['Close'] = df_multi['Close'] / n_days

    return df_multi

# df_com = merge_futures_contracts("CL")
# single_df = single_day_builder(df_com)
# df2 = multi_day_efficient(4, 17, single_df)
# print(df2)

# df = multi_day_roll(4, 17, df_com)
# print(df)

# exit()
# df_com = merge_futures_contracts(com_code)
# df = multi_day_roll(4, 17, df_com)
# print(df)
# print(get_mean_variance(df))
# exit()


rows = []
n_days = []
dte = []
for i in range(1, 25+1):
    for j in range(1, 25+1):
        rows = rows + ["{}-{}".format(i, j)]
        n_days = n_days + [i]
        dte = dte + [j]

# print(n_days)
# print(dte)

Com_codes = get_commodity_codes_df()
Com_codes = Com_codes['Code'].tolist()

df = pd.DataFrame(columns=Com_codes, index=rows)

Com_codes = Com_codes[4:15:-1]


for code in Com_codes:
    print()
    df_com = merge_futures_contracts(code)
    single_df = single_day_builder(df_com)
    for id in rows:
        index = rows.index(id)
        # print('{} - {}'.format(n_days[index], dte[index]))

        if index % 25 == 0:
            df.to_csv('{}\Multi_day_MV4.csv'.format(cwd))
            print(df)

        print('{} - {}%'.format(code, round(100*index/len(rows), 2)), end='\r')
        # temp_df = multi_day_roll(n_days[index], dte[index], df_com)
        temp_df = multi_day_efficient(n_days[index], dte[index], single_df)
        meanVar = get_mean_variance(temp_df)

        df.loc[id, "{}".format(code)] = meanVar


df.to_csv('{}\Multi_day_MV4.csv'.format(cwd))
