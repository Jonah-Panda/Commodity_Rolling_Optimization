import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

cwd = os.getcwd()

def rgbspectrum(x, min, diff):
    # red = [246, 52, 57]
    # gray = [65, 69, 84]
    # green = [48, 204, 90]
    if x == 0:
        return [255, 255, 255]

    pct = (x-min) / diff
    r = round(198*(1-pct)+48, 0)
    g = round(152*pct+52, 0)
    b = round(33*pct+57, 0)
    return [r, g, b]

def add_rect(x, y, rgb, ax2):
    ax2.add_patch(patches.Rectangle(
        (x, y), 
        1, 
        1,
        facecolor=(rgb[0]/255, rgb[1]/255, rgb[2]/255))) 
    return ax2

def show_code_plot(code, df):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    fig2.set_size_inches(3, 3.0)

    df2 = df.loc[:, ['{}'.format(code)]]
    mVar = df2['{}'.format(code)].tolist()
    mVar = [k for k in mVar if k != 0]
    df_max = max(mVar)
    df_min = min(mVar)
    df_diff = df_max - df_min

    for index, row in df2.iterrows():
        id = index
        id_split = id.split("-")
        n_days = int(id_split[0])
        end = int(id_split[1])

        rgb = rgbspectrum(row['{}'.format(code)], df_min, df_diff)
        
        ax2 = add_rect(n_days-1, end-1, rgb, ax2)
    
    return ax2


multi_df = pd.read_csv('Multi_day_MV_testing.csv', index_col=0)

training_df = pd.read_csv('Multi_day_MV_training.csv', index_col=0)

com_codes = multi_df.columns.tolist()
for com in com_codes:
    ax2 = show_code_plot(com, multi_df)
    ax2.set_xlim(0,25)
    ax2.set_ylim(0,25)
    ax2.set_xlabel("Days Rolled", fontsize=10)
    ax2.set_ylabel("Last Day Rolled", fontsize=10)
    ax2.set_title("{} - {}".format(com, "testing"))

    split = training_df['{}'.format(com)]
    split = split[split != 0]
    MVO_max = split.idxmax()
    [x, y] = MVO_max.split('-')

    ax2.add_patch(patches.Rectangle(
        (int(x)-1, int(y)-1), 
        1, 
        1,
        fill=False,
        ec='black'))

    ax2.add_patch(patches.Rectangle(
        (4-1, 5-1), 
        1, 
        1,
        fill=False,
        ec='yellow'))

    plt.tight_layout()
    plt.savefig('{}\{}'.format("Images_testing", com))
    plt.close()


# plt.show()