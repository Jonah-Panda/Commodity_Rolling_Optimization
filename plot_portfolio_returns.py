import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

STARTING_CAPITAL = 100
START_DATE = datetime.datetime(2020, 9, 8)
cwd = os.getcwd()


df = pd.read_csv('Returns.csv', index_col=0)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= START_DATE]
print(df)
# var_df = np.sqrt(df.loc[:, "TW_sam":"Comb_honey"].var()) * 100
# var_df.to_csv('{}\Images_plots\{}'.format(cwd, "sd_results.csv"))
# exit()

x = df['Date'].to_numpy()
strategies = df.columns.tolist()[1:-1]

plt.rcParams["figure.figsize"] = (8, 6)

for strat in strategies:
    if strat[0] == "W":
        continue
    elif strat[0] == "C" or strat[-1] == "":
        ret = df['{}'.format(strat)].to_numpy()
        ret[0] = ret[0] * STARTING_CAPITAL
        y = np.cumprod(ret)
        plt.plot(x, y, label='{}'.format(strat))

plt.legend(loc='lower right')
plt.title('Combined Minimum Variance $\delta$ Weighted (all methods)')
plt.ylabel('Return vs $100')
# plt.savefig('{}\Images_plots\{}'.format(cwd, "comb_plot"))
# plt.show()

plt.close()
plt.rcParams["figure.figsize"] = (12, 5)
plt.plot(x, df['delta_honey'].to_numpy(), label='$\delta$ honey')
plt.ylabel('$\delta$')
plt.title('$\delta$ Over Time')
plt.legend(loc='best')
plt.savefig('{}\Images_plots\{}'.format(cwd, "delta_honey"))
plt.show()
print(max(df['delta_honey']))
print(min(df['delta_honey']))