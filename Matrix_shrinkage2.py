import pandas as pd
import datetime
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np

START_DATE = datetime.datetime(2020, 9, 1)
concentration_ratio = 1/2

df = pd.read_csv('Two_week_ret_test.csv', index_col=0)
df['Date'] = pd.to_datetime(df['Date'])
id = df[df['Date'] < START_DATE].index[-1]+1
N_samples = int((len(df.columns)-1) / concentration_ratio)


df_slice = df.loc[id-N_samples:id, df.columns != 'Date']

df_cov = df_slice.cov()
# print(df_cov)
# Y = df_slice
# df_cov = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/40 
# print(df_cov)
# exit()

###########################################################################
def cov1Para(Y,k = None):
    
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

honey = cov1Para(df_slice)
# print(honey)

def covMarket(Y,k = None):
    
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

mkt = covMarket(df_slice)


eig = LA.eigvals(df_cov)
print(eig)
eig2 = LA.eigvals(honey)
print(eig2)
eig3 = LA.eigvals(mkt)
print(eig3)



plt.scatter(range(0, 21), eig, label='Raw')
plt.scatter(range(0, 21), eig2, label='honey')
plt.scatter(range(0, 21), eig3, label='mkt')
plt.legend(loc='upper right')
plt.show()