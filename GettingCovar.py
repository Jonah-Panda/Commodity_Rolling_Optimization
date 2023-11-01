# https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html

import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import os
import numpy as np

cwd = os.getcwd()
START_DATE = datetime.datetime(2020, 9, 1)
END_DATE = datetime.datetime(2023, 9, 1)

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
        # df['Volume_y'] = df['Volume_y'].fillna(0)

        roll_date = df.iloc[dte_roll, 0]
        new_contract_weight = df.loc[dte_roll, 'Close_x'] / df.loc[dte_roll, 'Close_y']
        # print(new_contract_weight)
        df['Weight1'] = np.where(df['Date'] < roll_date, 1, 0)
        df['Weight2'] = np.where(df['Date'] >= roll_date, new_contract_weight, 0)
        df['Close'] = df['Close_x'] * df['Weight1'] + df['Close_y'] * df['Weight2']
        # df['Volume'] = round(df['Volume_x'] * df['Weight1'] + df['Volume_y'] * df['Weight2'], 0)
        # print(df.iloc[dte_roll]) # checks that the contract roll is the same by dollars
        df = df[['Date', 'Close']]

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
        df = df[['Date', 'Close']]

    # Dividing cumulaitve sum by number of days to roll over
    df['Close'] = df['Close'] / n_days
    df = df[['Date', 'Close']]

    return df

datelist = pd.bdate_range(start=START_DATE - relativedelta(months=24), end=END_DATE)
df = pd.DataFrame(datelist, columns=['Date'])
del datelist

training_df = pd.read_csv('Multi_day_MV_training.csv', index_col=0)
com_codes = training_df.columns.tolist()

for com in com_codes:
    # Finding optimized roll strategy
    split = training_df['{}'.format(com)]
    split = split[split != 0]
    MVO_max = split.idxmax()
    [n_days, last_dte] = MVO_max.split('-')

    # Replacing HO since there are not enough overlapping days for contracts in the testing period
    if com == "HO":
        [n_days, last_dte] = [20, 20]

    # if com == "CL": ######################
    #     [n_days, last_dte] = [2, 2]#######

    # Getting Price History for testing period
    print("{}, - {}-{}".format(com, n_days, last_dte))
    df_com = multi_day_roll(int(n_days), int(last_dte), "{}".format(com))
    df_com['Date'] = pd.to_datetime(df_com['Date'])
    df_com.rename(columns={'Close':'{}'.format(com)}, inplace=True)
    df_com = df_com.loc[df_com['Date'] > START_DATE - relativedelta(months=24)]

    
    df = df.merge(df_com, on='Date', how='left')
    # if com == "CL": #############
    #     break ###################

df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
print(df)

df_weekly = df
df_weekly['DOW'] = df_weekly['Date'].dt.day_of_week
df_weekly['DOW_shift'] = df_weekly['DOW'].shift(-1)
df_weekly['EOW'] = df_weekly['DOW_shift'] - df_weekly['DOW']
df_weekly.loc[len(df_weekly)-1, 'EOW'] = -4
df_weekly = df_weekly[df_weekly['EOW'] < 0]
df_weekly = df_weekly.drop(['DOW', 'DOW_shift', "EOW"], axis=1)
df_weekly.reset_index(inplace=True, drop=True)

df_daily = df
df_daily = df_daily.drop(['DOW', 'DOW_shift', "EOW"], axis=1)

df_two_week = df_weekly[df_weekly.reset_index()['index'] % 2 == 0].reset_index(drop=True)

for com in com_codes:
    # two weeks
    df_pct = df_two_week.loc[:, ['Date', '{}'.format(com)]]
    df_pct.rename(columns={'{}'.format(com):'Close'}, inplace=True)
    df_pct['Close_minus_1'] = df_pct['Close'].shift(1)
    df_pct['Return'] = (df_pct['Close'] - df_pct['Close_minus_1']) / df_pct['Close_minus_1']
    df_two_week.loc[:, '{}'.format(com)] = df_pct['Return']

    # weekly
    df_pct = df_weekly.loc[:, ['Date', '{}'.format(com)]]
    df_pct.rename(columns={'{}'.format(com):'Close'}, inplace=True)
    df_pct['Close_minus_1'] = df_pct['Close'].shift(1)
    df_pct['Return'] = (df_pct['Close'] - df_pct['Close_minus_1']) / df_pct['Close_minus_1']
    df_weekly.loc[:, '{}'.format(com)] = df_pct['Return']

    # Daily
    df_pct = df_daily.loc[:, ['Date', '{}'.format(com)]]
    df_pct.rename(columns={'{}'.format(com):'Close'}, inplace=True)
    df_pct['Close_minus_1'] = df_pct['Close'].shift(1)
    df_pct['Return'] = (df_pct['Close'] - df_pct['Close_minus_1']) / df_pct['Close_minus_1']
    df_daily.loc[:, '{}'.format(com)] = df_pct['Return']

    # if com == "CL": ###########
    #     break #################


df_daily = df_daily[1:].reset_index(drop=True)
df_weekly = df_weekly[1:].reset_index(drop=True)
df_two_week = df_two_week[1:].reset_index(drop=True)

print(df_daily)
print(df_weekly)
print(df_two_week)

df_daily.to_csv('{}\Daily_ret_test.csv'.format(cwd))
df_weekly.to_csv('{}\Weekly_ret_test.csv'.format(cwd))
df_two_week.to_csv('{}\Two_week_ret_test.csv'.format(cwd))
