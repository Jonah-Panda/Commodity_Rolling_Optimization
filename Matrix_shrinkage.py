# https://github.com/pald22/covShrinkage
# Patrick has made the three covariance shrinkage functions
# Their github for the source of the functions is linked above
import pandas as pd
import datetime
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
import os

cwd = os.getcwd()
START_DATE = datetime.datetime(2020, 9, 1)
concentration_ratio = 1/2

def cov1Para(Y,k = None):
    # https://github.com/pald22/covShrinkage
    # Patrick has made the three covariance shrinkage functions
    # Their github for the source of the functions is linked above

    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import pandas as pd
    import math

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension
   
   
    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size
    
    
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    
    
    # compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar= sum(diag)/len(diag)
    target=meanvar*np.eye(p)
    
    
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    
    
    pihat = sum(piMat.sum())
    

    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    
    # diagonal part of the parameter that we call rho 
    rho_diag=0;
    
    # off-diagonal part of the parameter that we call rho 
    rho_off=0;
    
    # compute shrinkage intensity
    rhohat=rho_diag+rho_off
    kappahat=(pihat-rhohat)/gammahat
    shrinkage=max(0,min(1,kappahat/n))
    
    # compute shrinkage estimator
    sigmahat=shrinkage*target+(1-shrinkage)*sample
    
    
    return sigmahat
def covMarket(Y,k = None):
    # https://github.com/pald22/covShrinkage
    # Patrick has made the three covariance shrinkage functions
    # Their github for the source of the functions is linked above

    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import numpy.matlib as mt
    import pandas as pd
    import math

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension

    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size
    
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n

    #compute shrinkage target
    Ymkt = Y.mean(axis = 1) #equal-weighted market factor
    covmkt = pd.DataFrame(np.matmul(Y.T.to_numpy(),Ymkt.to_numpy()))/n #covariance of original variables with common factor
    varmkt = np.matmul(Ymkt.T.to_numpy(),Ymkt.to_numpy())/n #variance of common factor
    target = pd.DataFrame(np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy()))/varmkt
    target[np.logical_and(np.eye(p),np.eye(p))] = sample[np.logical_and(np.eye(p),np.eye(p))]
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    # diagonal part of the parameter that we call rho 
    rho_diag =  np.sum(np.diag(piMat))
    
    # off-diagonal part of the parameter that we call rho 
    temp = pd.DataFrame([Ymkt for i in range(p)]).T ############ Deleted Y* at the start of this before pd.DataFrame()
    covmktSQ = pd.DataFrame([covmkt[0] for i in range(p)])

    v1 = pd.DataFrame((1/n) * np.matmul(Y2.T.to_numpy(),temp.to_numpy())-np.multiply(covmktSQ.T.to_numpy(),sample.to_numpy()))
    roff1 = (np.sum(np.sum(np.multiply(v1.to_numpy(),covmktSQ.to_numpy())))-np.sum(np.diag(np.multiply(v1.to_numpy(),covmkt.to_numpy()))))/varmkt
    v3 = pd.DataFrame((1/n) * np.matmul(temp.T.to_numpy(),temp.to_numpy()) - varmkt * sample)
    roff3 = (np.sum(np.sum(np.multiply(v3.to_numpy(),np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy())))) - np.sum(np.multiply(np.diag(v3.to_numpy()),(covmkt[0]**2).to_numpy()))) /varmkt**2
    rho_off=2*roff1-roff3
    
    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = shrinkage*target + (1-shrinkage) * sample;
    
    return sigmahat
def covCor(Y,k = None):
    # https://github.com/pald22/covShrinkage
    # Patrick has made the three covariance shrinkage functions
    # Their github for the source of the functions is linked above
    
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned
    
    import numpy as np
    import numpy.matlib as mt
    import pandas as pd
    import math

    # de-mean returns if required
    N,p = Y.shape                      # sample size and matrix dimension
   
   
    #default setting
    if k is None or math.isnan(k):
        
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                    # adjust effective sample size

    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
        
    # compute shrinkage target
    samplevar = np.diag(sample.to_numpy())
    sqrtvar = pd.DataFrame(np.sqrt(samplevar))
    rBar = (np.sum(np.sum(sample.to_numpy()/np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy())))-p)/(p*(p-1)) # mean correlation
    target = pd.DataFrame(rBar*np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy()))
    target[np.logical_and(np.eye(p),np.eye(p))] = sample[np.logical_and(np.eye(p),np.eye(p))];
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    # diagonal part of the parameter that we call rho 
    rho_diag =  np.sum(np.diag(piMat))
    
    # off-diagonal part of the parameter that we call rho 
    term1 = pd.DataFrame(np.matmul((Y**3).T.to_numpy(),Y.to_numpy())/n)
    term2 = pd.DataFrame(np.transpose(mt.repmat(samplevar,p,1))*sample)
    thetaMat = term1-term2
    thetaMat[np.logical_and(np.eye(p),np.eye(p))] = pd.DataFrame(np.zeros((p,p)))[np.logical_and(np.eye(p),np.eye(p))]
    rho_off = rBar*(np.matmul((1/sqrtvar).to_numpy(),sqrtvar.T.to_numpy())*thetaMat).sum().sum()
    
    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = shrinkage*target + (1-shrinkage) * sample;
    
    return sigmahat
def slice_df(df_in_timeframe, start_date):
    df_in_timeframe['Date'] = pd.to_datetime(df_in_timeframe['Date'])

    id = df_in_timeframe[df_in_timeframe['Date'] < start_date].index[-1]+1
    N_samples = int((len(df_in_timeframe.columns)-1) / concentration_ratio)
    df_slice = df_in_timeframe.loc[id-N_samples:id, df_in_timeframe.columns != 'Date']
    return df_slice
def get_weights_reg(df_slice):
    # Calculating the covariance matrix
    df_cov = df_slice.cov()

    # Calculating the inverse covariance matrix
    df_cov_inv = pd.DataFrame(np.linalg.pinv(df_cov.values), df_cov.columns, df_cov.index)
    
    # Calculating the weight of each asset based on the diagonal of the inverse covariance matrix
    # df_cov_weight = np.diag(df_cov_inv.values) / sum(np.diag(df_cov_inv.values))
    df_cov_weight = np.sum(df_cov_inv.values,axis=0) / np.sum(df_cov_inv.values)

    return df_cov_weight
def get_weights_lsc(df_slice):
    # Calculating the covariance matrix
    lsc = cov1Para(df_slice)

    # Calculating the inverse covariance matrix
    lsc_inv = pd.DataFrame(np.linalg.pinv(lsc.values), lsc.columns, lsc.index)

    # Calculating the weight of each asset based on the diagonal of the inverse covariance matrix
    # lsc_weight = np.diag(lsc_inv.values) / sum(np.diag(lsc_inv.values))
    lsc_weight = np.sum(lsc_inv.values,axis=0) / np.sum(lsc_inv.values)

    return lsc_weight
def get_weights_mkt(df_slice):
    # Calculating the covariance matrix
    mkt = covMarket(df_slice)

    # Calculating the inverse covariance matrix
    mkt_inv = pd.DataFrame(np.linalg.pinv(mkt.values), mkt.columns, mkt.index)

    # Calculating the weight of each asset based on the diagonal of the inverse covariance matrix
    # mkt_weight = np.diag(mkt_inv.values) / sum(np.diag(mkt_inv.values))
    mkt_weight = np.sum(mkt_inv.values,axis=0) / np.sum(mkt_inv.values)
    return mkt_weight
def get_weights_honey(df_slice):
    # Calculating the covariance matrix
    honey = covCor(df_slice)

    # Calculating the inverse covariance matrix
    honey_inv = pd.DataFrame(np.linalg.pinv(honey.values), honey.columns, honey.index)

    # Calculating the weight of each asset based on the diagonal of the inverse covariance matrix
    # honey_weight = np.diag(honey_inv.values) / sum(np.diag(honey_inv.values))
    honey_weight = np.sum(honey_inv.values,axis=0) / np.sum(honey_inv.values)
    return honey_weight

# Getting price returns
df_TW = pd.read_csv('Two_week_ret_test.csv', index_col=0)
df_TW['Date'] = pd.to_datetime(df_TW['Date'])
df_W = pd.read_csv('Weekly_ret_test.csv', index_col=0)
df_W['Date'] = pd.to_datetime(df_W['Date'])
df_D = pd.read_csv('Daily_ret_test.csv', index_col=0)
df_D['Date'] = pd.to_datetime(df_D['Date'])

# Building portfolio
df = df_D.loc[:, ['Date']]
df['Date'] = df[df['Date'] >= START_DATE]#.reset_index(drop=True)
df.dropna(inplace=True)
df = df.reindex(columns=['Date', 'TW_sam', 'TW_mkt', 'TW_lsc', 'TW_honey', 'W_sam', 'W_mkt', 'W_lsc', 'W_honey', 'D_sam', 'D_mkt', 'D_lsc', 'D_honey', 'Comb_sam', 'Comb_mkt', 'Comb_lsc', 'Comb_honey', 'delta_honey'])
print(df)

df_trading_days_between = pd.DataFrame(df_D.loc[:, 'Date'])
df_TW['Day_count'] = np.ones(len(df_TW))
df_trading_days_between = df_trading_days_between.merge(df_TW.loc[:, ['Date', 'Day_count']], on='Date', how='left')
df_TW.drop(columns=['Day_count'], inplace=True)
df_trading_days_between = df_trading_days_between[df_trading_days_between['Date'] >= min(df_TW['Date'])].reset_index(drop=True)
df_trading_days_between = df_trading_days_between[df_trading_days_between['Day_count'] == 1].reset_index()
df_trading_days_between['index_t_minus_1'] = df_trading_days_between['index'].shift(-1)
df_trading_days_between['Days'] = df_trading_days_between['index_t_minus_1'] - df_trading_days_between['index']
df_trading_days_between.loc[len(df_trading_days_between)-1, 'Days'] = 10

rebalance_dates = df_TW['Date'].to_list()
trading_days_to_next_rebal = df_trading_days_between['Days'].tolist()
del df_trading_days_between

weights = np.zeros(len(df_D.columns)-1)
TW_sample_weights = weights
TW_mkt_weights = weights
TW_lsc_weights = weights
TW_honey_weights = weights
W_sample_weights = weights
W_mkt_weights = weights
W_lsc_weights = weights
W_honey_weights = weights
D_sample_weights = weights
D_mkt_weights = weights
D_lsc_weights = weights
D_honey_weights = weights
Comb_sample_weights = weights
Comb_mkt_weights = weights
Comb_lsc_weights = weights
Comb_honey_weights = weights
delta_honey = 0

for index, row in df.iterrows():
    date = row['Date']

    com_ret = df_D.loc[index, df_D.columns != 'Date'].to_numpy()

    TW_sam_ret = np.dot(1+com_ret, TW_sample_weights)
    TW_mkt_ret = np.dot(1+com_ret, TW_mkt_weights)
    TW_lsc_ret = np.dot(1+com_ret, TW_lsc_weights)
    TW_honey_ret = np.dot(1+com_ret, TW_honey_weights)
    W_sample_ret = np.dot(1+com_ret, W_sample_weights)
    W_mkt_ret = np.dot(1+com_ret, W_mkt_weights)
    W_lsc_ret = np.dot(1+com_ret, W_lsc_weights)
    W_honey_ret = np.dot(1+com_ret, W_honey_weights)
    D_sample_ret = np.dot(1+com_ret, D_sample_weights)
    D_mkt_ret = np.dot(1+com_ret, D_mkt_weights)
    D_lsc_ret = np.dot(1+com_ret, D_lsc_weights)
    D_honey_ret = np.dot(1+com_ret, D_honey_weights)
    Comb_sample_ret = np.dot(1+com_ret, Comb_sample_weights)
    Comb_mkt_ret = np.dot(1+com_ret, Comb_mkt_weights)
    Comb_lsc_ret = np.dot(1+com_ret, Comb_lsc_weights)
    Comb_honey_ret = np.dot(1+com_ret, Comb_honey_weights)


    df.loc[index, df.columns != 'Date'] = [TW_sam_ret, TW_mkt_ret, TW_lsc_ret, TW_honey_ret, W_sample_ret, W_mkt_ret, W_lsc_ret, W_honey_ret, D_sample_ret, D_mkt_ret, D_lsc_ret, D_honey_ret, Comb_sample_ret, Comb_mkt_ret, Comb_lsc_ret, Comb_honey_ret, delta_honey]

    if date in rebalance_dates:
        df_TW_slice = slice_df(df_TW, date)
        TW_sample_weights = get_weights_reg(df_TW_slice)
        TW_mkt_weights = get_weights_mkt(df_TW_slice)
        TW_lsc_weights = get_weights_lsc(df_TW_slice)
        TW_honey_weights = get_weights_honey(df_TW_slice)

        df_W_slice = slice_df(df_W, date)
        W_sample_weights = get_weights_reg(df_W_slice)
        W_mkt_weights = get_weights_mkt(df_W_slice)
        W_lsc_weights = get_weights_lsc(df_W_slice)
        W_honey_weights = get_weights_honey(df_W_slice)

        df_D_slice = slice_df(df_D, date)
        D_sample_weights = get_weights_reg(df_D_slice)
        D_mkt_weights = get_weights_mkt(df_D_slice)
        D_lsc_weights = get_weights_lsc(df_D_slice)
        D_honey_weights = get_weights_honey(df_D_slice)

        days_to_scale = trading_days_to_next_rebal[rebalance_dates.index(date)]
        twVol = np.sqrt(np.matmul(np.matmul(TW_sample_weights, df_TW_slice.cov()), np.transpose(TW_sample_weights)))
        dVol = np.sqrt(np.matmul(np.matmul(D_sample_weights, df_D_slice.cov()), np.transpose(D_sample_weights))) * np.sqrt(days_to_scale)
        delta = dVol / (twVol + dVol)
        Comb_sample_weights = D_sample_weights * (1 - delta) + TW_sample_weights * (delta)
        
        twVol = np.sqrt(np.matmul(np.matmul(TW_lsc_weights, cov1Para(df_TW_slice)), np.transpose(TW_lsc_weights)))
        dVol = np.sqrt(np.matmul(np.matmul(D_lsc_weights, cov1Para(df_D_slice)), np.transpose(D_lsc_weights))) * np.sqrt(days_to_scale)
        delta = dVol / (twVol + dVol)
        Comb_lsc_weights = D_lsc_weights * (1 - delta) + TW_lsc_weights * (delta)

        twVol = np.sqrt(np.matmul(np.matmul(TW_mkt_weights, covMarket(df_TW_slice)), np.transpose(TW_mkt_weights)))
        dVol = np.sqrt(np.matmul(np.matmul(D_mkt_weights, covMarket(df_D_slice)), np.transpose(D_mkt_weights))) * np.sqrt(days_to_scale)
        delta = dVol / (twVol + dVol)
        Comb_mkt_weights = D_mkt_weights * (1 - delta) + TW_mkt_weights * (delta)

        twVol = np.sqrt(np.matmul(np.matmul(TW_honey_weights, covCor(df_TW_slice)), np.transpose(TW_honey_weights)))
        dVol = np.sqrt(np.matmul(np.matmul(D_honey_weights, covCor(df_D_slice)), np.transpose(D_honey_weights))) * np.sqrt(days_to_scale)
        delta = dVol / (twVol + dVol)
        Comb_honey_weights = D_honey_weights * (1 - delta) + TW_honey_weights * (delta)
        delta_honey = delta

    else:
        continue
    continue

print(df)
df.to_csv('{}\Returns.csv'.format(cwd))









########################################################
# # print(rebalance_dates)
# date = datetime.datetime(2022, 3, 18)
# df_slice = slice_df(df_TW, date)
# df_cov = df_slice.cov()
# lsc = cov1Para(df_slice)
# mkt = covMarket(df_slice)
# honey = covCor(df_slice)

# print("Covariance Matricies")
# # print(df_cov)
# # print(lsc)
# # print(mkt)
# # print(honey)


# eig = LA.eigvals(df_cov)
# eig2 = LA.eigvals(lsc)
# eig3 = LA.eigvals(mkt)
# eig4 = LA.eigvals(honey)
# print("EigenValues")
# # print(eig)
# # print(eig2)
# # print(eig3)
# # print(eig4)

# df_cov_inv = pd.DataFrame(np.linalg.pinv(df_cov.values), df_cov.columns, df_cov.index)
# lsc_inv = pd.DataFrame(np.linalg.pinv(lsc.values), lsc.columns, lsc.index)
# mkt_inv = pd.DataFrame(np.linalg.pinv(mkt.values), mkt.columns, mkt.index)
# honey_inv = pd.DataFrame(np.linalg.pinv(honey.values), honey.columns, honey.index)

# df_cov_weight = np.diag(df_cov_inv.values) / sum(np.diag(df_cov_inv.values))
# lsc_weight = np.diag(lsc_inv.values) / sum(np.diag(lsc_inv.values))
# mkt_weight = np.diag(mkt_inv.values) / sum(np.diag(mkt_inv.values))
# honey_weight = np.diag(honey_inv.values) / sum(np.diag(honey_inv.values))

# print("Weights")
# # print(df_cov_weight)
# # print(lsc_weight)
# # print(mkt_weight)
# # print(honey_weight)

# Plotting Shrunken Eigenvalues
# plt.rcParams["figure.figsize"] = (4, 5)
# # plt.tight_layout()
# plt.scatter(range(0, 21), eig, label='Sample')
# plt.scatter(range(0, 21), eig2, label='LSC')
# # plt.scatter(range(0, 21), eig3, label='Mkt')
# # plt.scatter(range(0, 21), eig4, label='Honey')
# plt.legend(loc='upper right')
# plt.title('Eigenvalue Shrinkage (Sample vs LSC)')
# # plt.ylabel("$\lambda_i$")
# plt.xlabel("Index i of Eigenvalue")
# plt.savefig('{}\Images_plots\{}'.format(cwd, "lsc_vs_eigen"))
# plt.show()
