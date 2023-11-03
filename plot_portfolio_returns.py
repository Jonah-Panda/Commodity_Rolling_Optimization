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

x = df['Date'].to_numpy()
strategies = df.columns.tolist()[1:-1]

plt.rcParams["figure.figsize"] = (14, 8)

for strat in strategies:
    if strat[0] == "W":
        continue
    elif strat[0] == "" or strat[-1] == "y":
        ret = df['{}'.format(strat)].to_numpy()
        ret[0] = ret[0] * STARTING_CAPITAL
        y = np.cumprod(ret)
        plt.plot(x, y, label='{}'.format(strat))

plt.legend(loc='lower right')
plt.show()

plt.close()
plt.rcParams["figure.figsize"] = (8, 5)
plt.plot(x, df['delta_honey'].to_numpy(), label='$\delta$ honey')
plt.ylabel('$\delta$')
plt.title('$\delta$ Over Time')
plt.legend(loc='best')
plt.savefig('{}\Images_plots\{}'.format(cwd, "delta_honey"))
plt.show()