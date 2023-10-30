import pandas as pd
import os
import numpy as np
from numpy import where
from math import sqrt
import datetime

cwd = os.getcwd()
com_code = "CL"
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

df_com = merge_futures_contracts(com_code)

#start_weight = 1 ######

col_names = df_com.columns.tolist()[1:]
df = pd.DataFrame(columns=['Date', 'Close'])
df.loc[0, "Date"] = df_com['Date'][0]
df.loc[0, "Close"] = df_com['{}'.format(col_names[0])][0]

dte_roll = 5
old_contract_weight = 1
for i in range(len(col_names)-1):
    col = col_names[i]
    ltd = datetime.datetime(int(col[0:4]), int(col[5:7]), int(col[8:10]))
    df2 = df_com.loc[df_com["Date"] <= ltd, ["Date", "{}".format(col), "{}".format(col_names[i+1])]]
    df2.interpolate(method='pad', limit=10, inplace=True) # Padding for days with no trading volume

    roll_index = len(df2) - dte_roll
    roll_date = df2.loc[roll_index, ["Date"]][0]

    # HEY, IF YOU HAVE TIME YOU SHOULD TOTALLY COME BACK AND ADD VOLUME TRIGGERS AND CASH TRIGGERS TO THE MATCHING ALGO SO IT IS ACTUALLY REPRESENTATIVE
    new_contract_weight = df2.loc[df2['Date'] == roll_date, '{}'.format(col)] / df2.loc[df2['Date'] == roll_date, '{}'.format(col_names[i+1])]
    df2['Weight1'] = np.where(df2['Date'] < roll_date, old_contract_weight, 0)
    df2['Weight2'] = np.where(df2['Date'] >= roll_date, new_contract_weight, 0)
    old_contract_weight = new_contract_weight

    df2['Close'] = df2['{}'.format(col)] * df2['Weight1'] + df2['{}'.format(col_names[i+1])] * df2['Weight2']
    print(df2)
    df2 = df2[['Date', 'Close']]

    df2 = df2[df2['Date'] > df['Date'].max()]
    df = pd.concat([df, df2])
    
exit()
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
