import sys
import time
import os
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, Column 
from scipy.stats import linregress
from scipy import interpolate
from scipy import polyval, polyfit
from scipy import odr
import pylab as py
from matplotlib import gridspec
import sklearn.datasets as ds
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import corner
import emcee
import scipy.optimize as op
from scipy.linalg import cholesky, inv,det
from scipy.optimize import minimize
import random
from astropy.table import Table, Column
from linear_mcmc import *

from redTools import *
from Kcorrect import *

##################################################
band2 = 'w2'
inFile = '../reddening/ESN_HI_catal.csv'
scaler, pca = transform(inFile, band1 = 'r', band2 = 'w2')

u = scaler.mean_
s = scaler.scale_
v = scaler.var_
## z = (x-u)/s
##u: mean  s:scale  var=s**2

A = pca.explained_variance_ratio_              # The importance of different PCAs components

pca_inv_data = pca.inverse_transform(np.eye(3)) # coefficients to make PCs from features
p0 = pca_inv_data[0,0]
p1 = pca_inv_data[0,1]
p2 = pca_inv_data[0,2]

print u
print s
print p0, p1, p2
##################################################
def RFA_predict(features=['g_r', 'r_i', 'i_z', 'pc0'], out='i', max_depth=2000, n_estimators=2000, \
            min_samples_leaf=15, bootstrap=True):
    
    ## loading data for training the Random Forest algorithm
    inFile = '../reddening/ESN_INC_P0_MAG.csv'
    table_regr = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)


    ################################ Feature selection, regression parameters
#     features = ['g_r', 'r_i', 'i_z', 'pc0']
    output   = out + '_w1'
#     max_depth=2000
#     n_estimators=2000
    max_features=len(features)
#     min_samples_leaf=15
#     bootstrap=True
    ################################ Regression (Random Forest)

    regr_w1 = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, \
            max_features=max_features, min_samples_leaf=min_samples_leaf, \
                bootstrap=bootstrap)

    table_all = prepareSamples(table_regr, noTest=True)
    x_all, y_all = ML_data(table_all, features, output)
    regr_w1.fit(x_all, y_all)
    p_y_all_w1  = regr_w1.predict(x_all)
    d = np.abs(p_y_all_w1-y_all)
    tt = 0.6
    indx, = np.where(d<tt)
    X = y_all[indx]
    Xp = p_y_all_w1[indx]
    Y = X-Xp
    m1, b1 = np.polyfit(X,Y, 1)
    
    importances = regr_w1.feature_importances_
    print output, 'Feature Importances: ', importances
    ################################ Feature selection, regression parameters
    output   = out + '_w2'
    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, \
            max_features=max_features, min_samples_leaf=min_samples_leaf, \
                bootstrap=bootstrap)
    x_all, y_all = ML_data(table_all, features, output)
    regr.fit(x_all, y_all)
    p_y_all_w2  = regr.predict(x_all)
    d = np.abs(p_y_all_w2-y_all)
    tt = 0.6
    indx, = np.where(d<tt)
    X = y_all[indx]
    Xp = p_y_all_w2[indx]
    Y = X-Xp
    m2, b2 = np.polyfit(X,Y, 1)
    
    importances = regr.feature_importances_
    print output, 'Feature Importances: ', importances
    ################################ Regression (Random Forest)

#     print m1, b1
#     print m2, b2
    return regr_w1, regr, [m1,m2],[b1,b2]
##################################################
def predict_w1_w2(table, regr_w1, regr, m, b, features, output):
    a = time.time()

    W1_lst = []
    w1p_lst = []
    W2_lst = []
    w2p_lst = []

    for j in range(len(table['pgc'])):
        
        PGC = table['pgc'][j]
        W1  = table['w1'][j]
        W2  = table['w2'][j]
        INC = table["inc"][j]
        M21 = table["m21"][j]
        M0  = table["logWimx"][j]
        WBA = table["Wbap"][j]
        R50 = table["R50_w2p"][j]     
        
        if table['Sqlt'][j]>0:

            def f(w2):
                return predictor(w2, table, regr, features, output, index=j, m=m[1], b=b[1], useFullPredictions=True)
            ###############################################
            
            try: 
                out = solver(f, 7, 21, threshold=0.001) 
            except:
                out=None
            
            if out==None:
                print 'Not solvable ... !!!', PGC
                w2p = 0
                w1p = 0
            else:
                w2p, N = out
                w2p=w2p[0]       
                g_ = table['g0'][j]
                r_ = table['r0'][j]
                i_ = table['i0'][j]
                z_ = table['z0'][j]
                logWimx, c21w, mu50, P0 = get_PC(w2p, M21, M0, WBA, R50)
                g_-=redCorrect(INC, P0, band1='g', band2='w2')
                r_-=redCorrect(INC, P0, band1='r', band2='w2')
                i_-=redCorrect(INC, P0, band1='i', band2='w2')
                z_-=redCorrect(INC, P0, band1='z', band2='w2')
                outDict = {}
                outDict["g_r"] = g_-r_
                outDict["g_i"] = g_-i_
                outDict["g_z"] = g_-z_
                outDict["r_i"] = r_-i_
                outDict["r_z"] = r_-z_
                outDict["i_z"] = i_-z_ 
                
                outDict["pc0"] = P0
                inList=[]
                for f in features: inList.append(outDict[f])
                x_t = np.asarray([inList])
                i_w1p = (regr_w1.predict(x_t)+b[0])/(1.-m[0]) ### regr_w1.predict(x_t)  ### 
                w1p = i_-i_w1p[0]
        else:
            w2p = 0
            w1p = 0
        
        if table['Wqlt'][j]==0:
            W1=0
            W2=0

        W2_lst.append(W2)
        w2p_lst.append(w2p)
        W1_lst.append(W1)
        w1p_lst.append(w1p)    


        ###############################################

    W2_lst = np.asarray(W2_lst)
    w2p_lst = np.asarray(w2p_lst)
    W1_lst = np.asarray(W1_lst)
    w1p_lst = np.asarray(w1p_lst)


    print output, "Time: ", time.time()-a
    return W2_lst, w2p_lst, W1_lst, w1p_lst
##################################################
inFile = 'ESN_HI_catal_all.csv'

table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None, encoding=None)
table = extinctionCorrect(table)
table = Kcorrection(table)

##indx = np.asarray(range(650,670))
##table = trim(table, indx)

table['mu50'] = table[band2]+2.5*np.log10(2.*np.pi*(table['R50_'+band2]*60)**2)-2.5*np.log10(table['Wba'])
dWba2 = ((0.1/6./table['R50_'+band2])**2)*(1+table['Wba']**2)
c2 = (2.5/np.log(10))**2
table['Emu50']=np.sqrt(c2*(0.1/6./table['R50_'+band2])**2+c2*dWba2/table['Wba']**2+0.05**2)
table['EC82'] = (5*np.sqrt(2.)/np.log(10))*(0.1/6./table['R50_'+band2])

table['R50_w2p'] = halflight(table["R50_g"], table["R50_r"], table["R50_i"], table["R50_z"])
table['Wbap'] = ba(table["Sba"])

## u0, g0, r0, i0, z0 are already K-/extinction- corrected
table['u0'] = table['u']
table['g0'] = table['g']
table['r0'] = table['r']
table['i0'] = table['i']
table['z0'] = table['z']
table['w10']= table['w1']
table['w20']= table['w2']
table["scaler"] = scaler
table["Cpca"]   = pca
##################################################
print len(table['pgc'])

##################################################
###################### 0
band = str(sys.argv[1])
features=['g_r', 'r_i', 'i_z', 'pc0']
output = band+'_w2'
outName = 'corrected_mags_linewidth_all_'+band+'.csv'
if len(sys.argv) > 2:
    if band=='g':
        features=['r_i', 'i_z', 'pc0']
        output = 'r_w2'
        outName = 'corrected_mags_linewidth_all_g0.csv'
    if band=='r':
        features=['g_i', 'i_z', 'pc0']
        output = 'i_w2'
        outName = 'corrected_mags_linewidth_all_r0.csv'    
    if band=='i':
        features=['g_r', 'r_z', 'pc0']
        output = 'r_w2'
        outName = 'corrected_mags_linewidth_all_i0.csv'    
    if band=='z':
        features=['g_r', 'r_i', 'pc0']
        output = 'r_w2'
        outName = 'corrected_mags_linewidth_all_z0.csv'



regr_w1, regr, m, b = RFA_predict(features=features, out=band)
W2_lst, w2p_lst, W1_lst, w1p_lst = predict_w1_w2(table, regr_w1, regr, m, b, features, output)




myTable = Table()
myTable.add_column(Column(data=table['pgc'], name='pgc'))
myTable.add_column(Column(data=W1_lst, name='W1m_', format='%0.2f'))
myTable.add_column(Column(data=W2_lst, name='W2m_', format='%0.2f'))
myTable.add_column(Column(data=w1p_lst, name='W1p_', format='%0.2f'))
myTable.add_column(Column(data=w2p_lst, name='W2p_', format='%0.2f'))

myTable.write(outName, format='ascii.fixed_width',delimiter='|', bookend=False, overwrite=True) 


