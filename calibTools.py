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
from scipy.optimize import curve_fit
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

########################################################
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
########################################################

def linfit(x, m, b):
    return m * x + b
def linfit_b(x, m):
    return m * x

## When the slope is fixed, 'b' the intercept is variable
def linfit_m(x, b):
    return x+b

########################################################
def getReddening_err(table, band1='r', band2='w2'):
    
    pc0  = table['P0_w2']
    P1W2p = table['P0_w2p']
    R_source = table['R_source']
    indx, = np.where(R_source==0)
    pc0[indx] = P1W2p[indx]
    
    Epc0 = table['P0_w2_e']
    inc  = table['inc']   
    Einc = table['inc_e']
    
    if band1=='w2':
        a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1='w1', band2=band2)
        gamma = (a*pc0**3+b*pc0**2+c*pc0+d)
        d_gamma = np.abs((3*a*pc0**2+2*b*pc0+c)*Epc0)
        gamma = gamma/3.107
        d_gamma = gamma/3.107
    else:
        a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)
        gamma = (a*pc0**3+b*pc0**2+c*pc0+d)
        d_gamma = np.abs((3*a*pc0**2+2*b*pc0+c)*Epc0)
    
    indx, = np.where(gamma<0)
    gamma[indx]=0
    d_gamma[indx]=0

    q2 = 10**(-1.*theta)
    F = log_a_b(inc, q2)
    dF2 = Elogab2(inc, q2, Einc)
    dM2 = dF2*(a*pc0**3+b*pc0**2+c*pc0+d)**2+(F*(3*a*pc0**2+2*b*pc0+c)*Epc0)**2
    
    Aj_e2 = np.sqrt(dM2)
    Aj = F*gamma
    
    return Aj_e2

########################################################
def makeCluster(table, band='i', reject=[], weird=[], clusterName='', 
               nest='NEST_100001', isVirgo=False, slope=None, 
               pgcFaint=[], magCorrection=None, OP_IR=False):
    
    ctl   = np.genfromtxt(nest+'.csv' , delimiter='|', filling_values=-1, 
                          names=True, dtype=None, encoding=None)
    PGC = ctl['PGC']
    
    pgc       = table['pgc']
    logWimx   = table['logWimx']
    logWimx_e = table['logWimx_e']
    Vhel      = table['Vhel']
    inc       = table['inc']
    Sqlt      = table['Sqlt']

    Aj_e2     = getReddening_err(table, band1=band, band2='w2')


    mag = table[band+'_sss']
    
    if not magCorrection is None:
        mag = mag + magCorrection
    
    
    mag_e = np.sqrt((mag*0.+0.05)**2+Aj_e2**2)    
    
    pgc_ = []
    logWimx_ = []
    logWimx_e_ = []
    mag_ = []
    mag_e_ = []
    Sqlt_ = []
    Wqlt_ = []
    
    if band=='u': 
        reject.append(40105)
    if band=='z': 
        reject.append(42089)    
    
    for i, idd in enumerate(pgc):
        if idd in PGC and not idd in reject:# and inc[i]>0. and Sqlt[i]>1:
            
            if (OP_IR==True and table['Sqlt'][i]>=2 and table['Wqlt'][i]>=2) or OP_IR==False:
                if isVirgo:
                    if Vhel[i]<600 or (Vhel[i]>1200 and Vhel[i]<1600):
                        if mag[i]>5 and mag[i]<20:
                            pgc_.append(idd)
                            logWimx_.append(logWimx[i])
                            logWimx_e_.append(logWimx_e[i])
                            mag_.append(mag[i])
                            mag_e_.append(mag_e[i])
                            Sqlt_.append(table['Sqlt'][i])
                            Wqlt_.append(table['Wqlt'][i])
                else:
                     if mag[i]>5 and mag[i]<20:
                        pgc_.append(idd)
                        logWimx_.append(logWimx[i])
                        logWimx_e_.append(logWimx_e[i])
                        mag_.append(mag[i])
                        mag_e_.append(mag_e[i])
                        Sqlt_.append(table['Sqlt'][i])
                        Wqlt_.append(table['Wqlt'][i])                
                
            
    pgc_=np.asarray(pgc_)
    logWimx_=np.asarray(logWimx_)
    logWimx_e_=np.asarray(logWimx_e_)
    mag_=np.asarray(mag_)
    mag_e_=np.asarray(mag_e_)
    Sqlt_=np.asarray(Sqlt_)
    Wqlt_=np.asarray(Wqlt_)
    N = len(pgc_)
    dofit = np.zeros(N)
      
    for i in range(N):
        if not pgc_[i] in weird:
            if not band in ['w1','w2'] and Sqlt_[i]>=2:
                dofit[i]=1
            elif band in ['w1','w2'] and Wqlt_[i]>=2:
                dofit[i]=1
            else:
                dofit[i]=2   # bad quality 
        else:
            dofit[i]=2   # already in the weird list
        if pgc_[i] in pgcFaint and not pgc_[i] in weird:   # faint Imag < -17
            dofit[i]=3   
        
    indx, = np.where(dofit==2)
    pgc_w = pgc_[indx]
    logWimx_w= logWimx_[indx]
    logWimx_e_w= logWimx_e_[indx]
    mag_w= mag_[indx]
    mag_e_w= mag_e_[indx]
    
    indx, = np.where(dofit==1) # actual fit
    pgc_f = pgc_[indx]
    logWimx_f= logWimx_[indx]
    logWimx_e_f= logWimx_e_[indx]
    mag_f= mag_[indx]
    mag_e_f= mag_e_[indx]
    
    indx, = np.where(dofit!=2)   # not weird (either bright or faint)
    pgc_ = pgc_[indx]
    logWimx_= logWimx_[indx]
    logWimx_e_= logWimx_e_[indx]
    mag_= mag_[indx]
    mag_e_= mag_e_[indx]
        
    
    
    try:
        ### 'Virgo W cluster' condition
        if band in ['u', 'g']:
            indx = np.asarray([int(i) for i, idd in enumerate(pgc_f) if not idd in [39886, 39114]])
            logWimx_0= logWimx_f[indx]
            logWimx_e_0= logWimx_e_f[indx]
            mag_0= mag_f[indx]
            mag_e_0= mag_e_f[indx]     

            if slope is None:
                fit, cov = curve_fit(linfit, mag_0, logWimx_0-2.5, sigma=np.sqrt(logWimx_e_0**2+0.01**2))
            else:
                fit, cov = curve_fit(linfit_m, mag_0/slope, logWimx_0-2.5, sigma=np.sqrt(logWimx_e_0**2+0.01**2))
            
        else:
            if slope is None :
                fit, cov = curve_fit(linfit, mag_f, logWimx_f-2.5, sigma=np.sqrt(logWimx_e_f**2+0.01**2))
            else:
                fit, cov = curve_fit(linfit_m, mag_f/slope, logWimx_f-2.5, sigma=np.sqrt(logWimx_e_f**2+0.01**2))
            
        if slope is None :
            m, b = fit
            me = np.sqrt(cov[0,0])
            be = np.sqrt(cov[1,1])
            mbe = cov[1,0]    
            m0 = 1./m
            b0 = -b/m
            dm0 = np.abs(me/m**2)
            db0 = np.sqrt((b0**2 )*(be**2/b**2+me**2/m**2-2*mbe/m/b))
        else:
            b = fit[0]
            me = 0
            be = np.sqrt(cov[0])
            m0 = slope
            b0 = -b*slope
            dm0 = 0
            db0 = np.abs(slope*be)          
    except:
        m0,b0,dm0,db0 = 0,0,0,0
        'Warning ....'
    
    
    ## '_w' stands for weird
    outDict = {'pgc':pgc_, 'slope':m0, 'zp':b0, 'slope_e':dm0, 'zp_e':db0, 
               'nest':nest, 'name':clusterName, 'band':band, 
               'mag':mag_, 'mag_e':mag_e_, 'logWimx':logWimx_, 'logWimx_e':logWimx_e_,
              'pgc_w':pgc_w, 'mag_w':mag_w, 'mag_e_w':mag_e_w, 'logWimx_w':logWimx_w, 
               'logWimx_e_w':logWimx_e_w, 'reject':reject, 'weird':weird}
    
    return outDict

########################################################
def allCluster(table, band='i', slope=None, pgcFaint=[], magCorrection=None,
               addSouth=True, OP_IR=False):
    
    Clusters = {}

    reject = [43164,44405,93666]
    weird = [43511]
    myDict = makeCluster(table, nest='NEST_100001', clusterName='Coma', 
                        reject=reject, weird=weird, band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_100001'] = myDict

    reject = []
    weird = [41440, 40809]
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Virgo', nest='NEST_100002', isVirgo=True, band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_100002'] = myDict


    ### SOUTH
    if addSouth:
        reject = []
        weird = []
        myDict = makeCluster(table, reject=reject, weird=weird, 
                   clusterName='Centaurus', nest='NEST_100003', band=band, slope=slope, pgcFaint=pgcFaint, 
                            magCorrection=magCorrection, OP_IR=OP_IR)
        Clusters['NEST_100003'] = myDict

    reject = [36323,36328,36330,36608,200155]
    weird = [37140]
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Abell 1367', nest='NEST_100005', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_100005'] = myDict


    ### SOUTH
    if addSouth:
        reject = []
        weird = [31500]
        myDict = makeCluster(table, reject=reject, weird=weird, 
                   clusterName='Hydra', nest='NEST_100006', band=band, slope=slope, pgcFaint=pgcFaint, 
                            magCorrection=magCorrection, OP_IR=OP_IR)
        Clusters['NEST_100006'] = myDict

    reject = [56977,2790835]
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Abell 2151 (Hercules)', nest='NEST_100007', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_100007'] = myDict


    reject = [37550]
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Ursa Major', nest='NEST_100008', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_100008'] = myDict

    ### SOUTH
    if addSouth:
        reject = []
        weird = []
        myDict = makeCluster(table, reject=reject, weird=weird, 
                   clusterName='Antlia', nest='NEST_100014', band=band, slope=slope, pgcFaint=pgcFaint, 
                            magCorrection=magCorrection, OP_IR=OP_IR)
        Clusters['NEST_100014'] = myDict

    reject = [38333]
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='NGC4065', nest='NEST_100018', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_100018'] = myDict

    reject = [23308]
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Cancer', nest='NEST_100030', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_100030'] = myDict

    #reject = [39655] 
    #weird = []
    #myDict = makeCluster(table, reject=reject, weird=weird, 
               #clusterName='Virgo W', nest='NEST_120002', band=band, slope=slope, pgcFaint=pgcFaint, 
                        #magCorrection=magCorrection, OP_IR=OP_IR)
    #Clusters['NEST_120002'] = myDict

    reject = [] 
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Abell 262', nest='NEST_200003', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200003'] = myDict

    reject = [3446,4020] 
    weird = [1904373]
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='NGC410', nest='NEST_200005', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200005'] = myDict

    reject = [4740,4876,5008] 
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='NGC507', nest='NEST_200006', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200006'] = myDict

    ### SOUTH
    if addSouth:
        reject = [] 
        weird = []
        myDict = makeCluster(table, reject=reject, weird=weird, 
                   clusterName='Fornax', nest='NEST_200015', band=band, slope=slope, pgcFaint=pgcFaint, 
                            magCorrection=magCorrection, OP_IR=OP_IR)
        Clusters['NEST_200015'] = myDict

    reject = [11150,11199,138562,3647754] 
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Abell 400', nest='NEST_200012', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200012'] = myDict

    reject = [85526,85643,90431,197699] 
    weird = [5057398]
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Abell 2634/66', nest='NEST_200016', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200016'] = myDict

    ### SOUTH
    if addSouth:
        reject = [] 
        weird = []
        myDict = makeCluster(table, reject=reject, weird=weird, 
                   clusterName='Abell 539', nest='NEST_200017', band=band, slope=slope, pgcFaint=pgcFaint, 
                            magCorrection=magCorrection, OP_IR=OP_IR)
        Clusters['NEST_200017'] = myDict

    reject = [1724] 
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='NGC70', nest='NEST_200037', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200037'] = myDict

    reject = [90474] 
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='NGC80', nest='NEST_200045', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200045'] = myDict

    reject = [70712, 70998, 71360, 71097] 
    weird = []
    myDict = makeCluster(table, reject=reject, weird=weird, 
               clusterName='Pegasus', nest='NEST_200092', band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    Clusters['NEST_200092'] = myDict
    
    return Clusters

########################################################
def makeFig(band='i', xLim = (1.5,2.9), yLim = (19.5,6.5), MAGabs=False, getTwinAX=False):
    
    fig = py.figure(figsize=(5,5), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.9, bottom=0.12, left=0.05, right=0.98)
    ax = fig.add_subplot(111)
    ax.set_xlim(xLim)        
    ax.set_ylim(yLim)

    if MAGabs: # absolute magnitudes
        if not band in ['w1','w2']:
            ax.set_ylabel(r'$M^*_{'+band+'} \/\/ [AB]$', fontsize=18) 
        else:
            ax.set_ylabel(r'$M^*_{'+band.upper()+'} \/\/ [AB]$', fontsize=18)
        ax.set_xlabel(r'$Log( W_{mx}^i)$', fontsize=18)         
    else:    # apparent magnitudes
        if not band in ['w1','w2']:
            ax.set_ylabel(r'$'+band+'^* \/\/ [AB]$', fontsize=18) 
        else:
            ax.set_ylabel(r'$'+band.upper()+'^* \/\/ [AB]$', fontsize=18)
        ax.set_xlabel(r'$Log( W_{mx}^i)$', fontsize=18) 
    
    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    
    ax.set_xticks(np.arange(2,3.5, step=0.5))

    if True:
        y_ax = ax.twinx()
        y_ax.set_ylim(yLim)
        y_ax.set_yticklabels([])
        y_ax.minorticks_on()
        y_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')        

    if True:
        x_ax = ax.twiny()
        x_ax.set_xlim(xLim)
        x_ax.set_xticklabels([])
        x_ax.minorticks_on()
        x_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')     
        x_ax.set_xticks(np.arange(2, 3.5, step=0.5))


    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(14) 
            
    if not getTwinAX:
        return fig, ax
    else:
        return fig, ax, x_ax, y_ax

########################################################
## is nest is None --> we are dealing with zp calibrators
def plotCluster(ax, Clusters, nest=None, 
                color=None, plotWeird=False, offset=0, 
                plotFit=False, alpha=0.7, symbol='o', 
                plotErrors=False, markersize=4, facecolor=None, 
                pgcFaint=[], alphaFaint=0.1, weirdColor='k', alphaWeird=0.2):
    
    if nest is not None:
        myCluster = Clusters[nest]
        mag_      = myCluster['mag']
        mag_w     = myCluster['mag_w']
    else:
        myCluster = Clusters
        mag_      = myCluster['MAG']
        mag_w     = myCluster['MAG_w']   

    pgc = myCluster['pgc']
    band = myCluster['band']   
    logWimx_  = myCluster['logWimx']
    logWimx_e_= myCluster['logWimx_e']
    mag_e_    = myCluster['mag_e']
    

    indx = []
    indx_ = []    
    for i, id in enumerate(pgc):
        if not id in pgcFaint:
            indx.append(i)
        else:
            indx_.append(i)        

    mag_g = mag_
    mag_e_g = mag_e_
    logWimx_g = logWimx_
    logWimx_e_g = logWimx_e_
    if len(indx_)>0:
        indx_ = np.asarray(indx_)
        mag_g = mag_[indx_]
        mag_e_g = mag_e_[indx_]
        logWimx_g = logWimx_[indx_]
        logWimx_e_g = logWimx_e_[indx_]        
    if len(indx)>0:
        indx = np.asarray(indx)
        mag_ = mag_[indx]
        mag_e_ = mag_e_[indx]
        logWimx_ = logWimx_[indx]
        logWimx_e_ = logWimx_e_[indx]
    
    
    logWimx_w  = myCluster['logWimx_w']
    logWimx_e_w= myCluster['logWimx_e_w']
    mag_e_w   = myCluster['mag_e_w']


    slope = myCluster['slope']
    zp = myCluster['zp']
    
    if slope==0: return

    dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple", 'w2': "brown"}
    if color==None: color = dye[band]
    if offset==None:
        offset = zp

    if plotWeird:
        if plotErrors:
            ax.errorbar(logWimx_w, mag_w-offset, 
                        xerr=logWimx_e_w, yerr=mag_e_w, 
                        fmt=symbol, color=weirdColor, 
                        markersize=4, markerfacecolor='white', alpha=alphaWeird) 
        else:
            ax.errorbar(logWimx_w, mag_w-offset, 
                        xerr=logWimx_e_w*0, yerr=mag_e_w*0, 
                        fmt=symbol, color=weirdColor, 
                        markersize=4, markerfacecolor='white', alpha=alphaWeird)             
    
    if plotErrors:
        ax.errorbar(logWimx_, mag_-offset, 
                    xerr=logWimx_e_, yerr=mag_e_, 
                    fmt=symbol, color=color, markersize=4, alpha=alpha, markerfacecolor=facecolor)

        if len(indx_)>0:
            ax.errorbar(logWimx_g, mag_g-offset, 
                    xerr=logWimx_e_g, yerr=mag_e_g, 
                    fmt=symbol, color='k', markersize=4, alpha=alphaFaint, markerfacecolor=facecolor)     
    else:
        ax.errorbar(logWimx_, mag_-offset, 
                    xerr=logWimx_e_*0, yerr=mag_e_*0, 
                    fmt=symbol, color=color, markersize=4, alpha=alpha, markerfacecolor=facecolor)        
        if len(indx_)>0:
            ax.errorbar(logWimx_g, mag_g-offset, 
                    xerr=logWimx_e_g*0, yerr=mag_e_g*0, 
                    fmt=symbol, color='k', markersize=4, alpha=0.9, markerfacecolor='white')      

    if plotFit:
        x = np.linspace(-5,5,50)
        y = slope*(x-2.5)+zp-offset
        ax.plot(x,y, 'k--')  
        
        x0 = 2.5
        y0 = 100
        x1 = 2.5
        y1 = slope*(x1-2.5)+zp-offset
        ax.plot([x0,x1],[y0,y1], ':', color='maroon') 
        
        x2 = -5
        y2 = y1
        x3 = x1
        y3 = y1
        ax.plot([x2,x3],[y2,y3], ':', color='maroon')       
   
    return logWimx_, mag_, logWimx_e_, mag_e_

########################################################
### 'NEST_100002' is the Virgo cluster, we take it as reference
def TF_iter(table, key0 = 'NEST_100002', band = 'i', 
            n_iter=10, pgcFaint=[], verbose=False, magCorrection=None,
               addSouth=True, OP_IR=False):
    
    Clusters = allCluster(table, band=band, slope=None, pgcFaint=pgcFaint, magCorrection=magCorrection, addSouth=addSouth, OP_IR=OP_IR)
    
    for repeat in range(n_iter):
        myCluster = Clusters[key0]
        zp = myCluster['zp']
        mag = myCluster['mag']-zp
        mag_e = myCluster['mag_e']
        logWimx = myCluster['logWimx']
        logWimx_e = myCluster['logWimx_e']

        for i, key in enumerate(Clusters):
            if key!=key0:
                myCluster = Clusters[key]
                zp = myCluster['zp']
                mag = np.concatenate((mag, myCluster['mag']-zp))
                mag_e = np.concatenate((mag_e, myCluster['mag_e']))
                logWimx = np.concatenate((logWimx, myCluster['logWimx']))
                logWimx_e = np.concatenate((logWimx_e, myCluster['logWimx_e']))


        fit, cov = curve_fit(linfit, mag, logWimx-2.5, sigma=np.sqrt(logWimx_e**2+0.01**2))
        m, b = fit
        me = np.sqrt(cov[0,0])
        slope = 1./m
        zp = -b/m   # theoretically zp should equal 0 at this point, since all magnitudes were shifted
        slope_e = np.abs(me/m**2)

        Clusters = allCluster(table, band=band, slope=slope, pgcFaint=pgcFaint, magCorrection=magCorrection, addSouth=addSouth, OP_IR=OP_IR)

        if verbose:
            print repeat, slope, zp

    if verbose: 
        for key in Clusters:
            print key, Clusters[key]['name'], "%.2f"%Clusters[key]['slope'], "%.2f"%Clusters[key]['zp']
    
    return Clusters, slope, zp, slope_e
########################################################
def makeZP(table, band='i', reject=[], weird=[], clusterName='', nest='', slope=None, pgcFaint=[], magCorrection=None, OP_IR=False):
    
    ctl   = np.genfromtxt('zp_photom_reduced.csv' , delimiter='|', filling_values=-1, 
                          names=True, dtype=None, encoding=None)
    PGC  = ctl['PGC']
    ID   = ctl['Name']
    dist = ctl['d']
    
    pgc       = table['pgc']
    logWimx   = table['logWimx']
    logWimx_e = table['logWimx_e']
    Vhel      = table['Vhel']

    Aj_e2     = getReddening_err(table, band1=band, band2='w2')

    
    mag = table[band+'_sss']
    if not magCorrection is None:
           mag = mag + magCorrection

    mag_e = np.sqrt((mag*0.+0.05)**2+Aj_e2**2)    
    
    pgc_      = []
    logWimx_  = []
    logWimx_e_= []
    mag_      = []
    mag_e_    = []
    Sqlt_     = []
    Wqlt_     = []
    dist_     = []
    ID_       = []
    
    for i, idd in enumerate(pgc):
        if idd in PGC and not idd in reject:
            if (OP_IR==True and table['Sqlt'][i]>=2 and table['Wqlt'][i]>=2) or OP_IR==False:
                indx, = np.where(PGC==idd)
                if mag[i]>5 and mag[i]<22:
                        pgc_.append(idd)
                        logWimx_.append(logWimx[i])
                        logWimx_e_.append(logWimx_e[i])
                        mag_.append(mag[i])
                        mag_e_.append(mag_e[i])
                        Sqlt_.append(table['Sqlt'][i])
                        Wqlt_.append(table['Wqlt'][i])     
                        dist_.append(dist[indx][0])
                        ID_.append(ID[indx][0])
            
            
    pgc_=np.asarray(pgc_)
    logWimx_=np.asarray(logWimx_)
    logWimx_e_=np.asarray(logWimx_e_)
    mag_=np.asarray(mag_)
    mag_e_=np.asarray(mag_e_)
    Sqlt_=np.asarray(Sqlt_)
    Wqlt_=np.asarray(Wqlt_)
    dist_=np.asarray(dist_)
    ID_=np.asarray(ID_)
    N = len(pgc_)
    dofit = np.zeros(N)
      
    for i in range(N):
        if not pgc_[i] in weird:
            if not band in ['w1','w2'] and Sqlt_[i]>=2:
                dofit[i]=1
            elif band in ['w1','w2'] and Wqlt_[i]>=2:
                dofit[i]=1
            else:
                dofit[i]=2
        else:
            dofit[i]=2
        if pgc_[i] in pgcFaint and not pgc_[i] in weird:   # faint Imag < -17
            dofit[i]=3   
                        
    indx,       = np.where(dofit==2)
    pgc_w       = pgc_[indx]
    logWimx_w   = logWimx_[indx]
    logWimx_e_w = logWimx_e_[indx]
    mag_w       = mag_[indx]
    mag_e_w     = mag_e_[indx]
    dist_w      = dist_[indx]
    ID_w        = ID_[indx]
    MAG_w       = mag_w - 5*np.log10(dist_w) - 25.
    
    indx,       = np.where(dofit==1)
    pgc_f        = pgc_[indx]
    logWimx_f    = logWimx_[indx]
    logWimx_e_f  = logWimx_e_[indx]
    mag_f        = mag_[indx]
    mag_e_f      = mag_e_[indx]
    dist_f       = dist_[indx]
    ID_f         = ID_[indx]
    MAG_f        = mag_f - 5*np.log10(dist_f) - 25.


    indx,       = np.where(dofit!=2)
    pgc_        = pgc_[indx]
    logWimx_    = logWimx_[indx]
    logWimx_e_  = logWimx_e_[indx]
    mag_        = mag_[indx]
    mag_e_      = mag_e_[indx]
    dist_       = dist_[indx]
    ID_         = ID_[indx]
    MAG_        = mag_ - 5*np.log10(dist_) - 25.
    
    try:
        
        if slope is None :
                fit, cov = curve_fit(linfit, MAG_f, logWimx_f-2.5, sigma=np.sqrt(logWimx_e_f**2+0.01**2))
                
                m, b = fit
                me = np.sqrt(cov[0,0])
                be = np.sqrt(cov[1,1])
                mbe = cov[1,0]    
                m0 = 1./m
                b0 = -b/m
                dm0 = np.abs(me/m**2)
                db0 = np.sqrt((b0**2 )*(be**2/b**2+me**2/m**2-2*mbe/m/b))                
        else:
                fit, cov = curve_fit(linfit_m, MAG_f/slope, logWimx_f-2.5, sigma=np.sqrt(logWimx_e_f**2+0.01**2))
                
                b = fit[0]
                me = 0
                be = np.sqrt(cov[0])
                m0 = slope
                b0 = -b*slope
                dm0 = 0
                db0 = np.abs(slope*be)          
    except:
        m0,b0,dm0,db0 = 0,0,0,0
        'Warning ....'
        
   
    outDict = {'pgc':pgc_, 'slope':m0, 'zp':b0, 'slope_e':dm0, 'zp_e':db0, 
               'nest':nest, 'name':clusterName, 'band':band, 
               'mag':mag_, 'mag_e':mag_e_, 'logWimx':logWimx_, 'logWimx_e':logWimx_e_,
              'pgc_w':pgc_w, 'mag_w':mag_w, 'mag_e_w':mag_e_w, 'logWimx_w':logWimx_w, 
               'logWimx_e_w':logWimx_e_w, 'reject':reject, 'weird':weird, 'MAG':MAG_, 'MAG_w':MAG_w,
              'dist':dist_, 'dist_w':dist_w, 'ID':ID_, 'ID_w':ID_w}
    
    return outDict

########################################################
def all_ZP(table, band='i', slope=None, pgcFaint=[], magCorrection=None, OP_IR=False):
    
    reject = []
    
    ## remove 42407 (sombrero) ? 
    if not band in ['w1','w2']:
        weird = [68535, 28378, 42081, 39461, 44536]
    else:
        weird = [68535, 28378, 42081, 39461, 44536, 21102,50073, 5896, 15345]
    
    myDict = makeZP(table, reject=reject, weird=weird, 
                    band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    
    return myDict
########################################################
def makeTF(table, pgcFaint=[], calib_maglim=[], band='i', makePlot=False, 
                        magCorrection=None, addSouth=True, getZPcalib=False, OP_IR=False):
    
    Clusters, slope0, zp0, slope_e0 = TF_iter(table, band = band, n_iter=10, 
                                              verbose=False, pgcFaint=pgcFaint, 
                                              magCorrection=magCorrection, addSouth=addSouth, OP_IR=OP_IR)

    zp_calibs = all_ZP(table, band=band, slope=slope0, pgcFaint=calib_maglim, 
                        magCorrection=magCorrection, OP_IR=OP_IR)
    slope = zp_calibs['slope']
    zp    = zp_calibs['zp']
    zp_e = zp_calibs['zp_e']
    
    Clusters = allCluster(table, band=band, slope=slope, pgcFaint=pgcFaint, 
                        magCorrection=magCorrection, addSouth=addSouth, OP_IR=OP_IR)
    
    if not makePlot:
        return Clusters, np.asarray([slope0, slope_e0, zp, zp_e]), zp_calibs
    else:
        
        fig, ax = makeFig(band=band, xLim = (1.5,3.1), yLim = (-12,-24), MAGabs=True)

        key0 = 'NEST_100002'
        myCluster = Clusters[key0]
        zp_virgo = myCluster['zp']
        
        N_cluster = 0

        for i, key in enumerate(Clusters):
            
            myCluster = Clusters[key]
            
            pgc =  myCluster['pgc']
            indx = []
            indx_ = []    
            for i, id in enumerate(pgc):
                if not id in pgcFaint:
                    indx.append(i)
                else:
                    indx_.append(i)  
            N_cluster += len(pgc[indx])
            
            if key!=key0:
                zp_ = myCluster['zp']
                plotCluster(ax, Clusters, nest=key, offset=zp_-zp, plotErrors=True, 
                            alpha=0.2, markersize=0, pgcFaint=pgcFaint, plotWeird=True) # cmap(i)
            else:
                plotCluster(ax, Clusters, nest=key, offset=zp_virgo-zp, plotErrors=True, alpha=0.2, 
                            markersize=0, pgcFaint=pgcFaint, plotWeird=True)
        
        ## plotting distance calibrators
        plotCluster(ax, zp_calibs, markersize=8, plotFit=True, color='k', pgcFaint=calib_maglim)
        
        pgc = zp_calibs['pgc']
        indx = []
        indx_ = []    
        for i, id in enumerate(pgc):
            if not id in calib_maglim:
                indx.append(i)
            else:
                indx_.append(i)    
        N_calib = len(pgc[indx])

        ax.text(2.1,-14, "%d" % N_cluster+' Cluster Galaxies', fontsize=12, color='k')
        ax.text(2.1,-13, "%d" % N_calib+' Zeropoint Galaxies', fontsize=12, color='k')
    
        Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
        x0 = 0.95*Xlm[0]+0.05*Xlm[1]
        y0 = 0.1*Ylm[0]+0.90*Ylm[1]
        ax.text(x0,y0, "Slope = "+"%.2f" % slope0+'$\pm$'+"%.2f" % slope_e0, fontsize=12, color='k')
        y0 = 0.2*Ylm[0]+0.80*Ylm[1]
        ax.text(x0,y0, "ZP = "+"%.2f" % zp+'$\pm$'+"%.2f" % zp_e, fontsize=12, color='k')
        
        if not getZPcalib:
            return fig, ax, Clusters, np.asarray([slope0, slope_e0, zp, zp_e])
        else:
            return fig, ax, Clusters, np.asarray([slope0, slope_e0, zp, zp_e]), zp_calibs
########################################################
def LFfunction(M, Ms, alpha):
    dm = 0.4*(Ms-M)
    c = 10.**dm
    b = c**(alpha+1)
    
    return b/np.exp(c)

########################################################
def Normal(x, mu, sigma):
    
    y = np.exp(-(x-mu)**2/(2.*sigma**2))
    
    return y/sigma/np.sqrt(2.*np.pi)
########################################################
def simulFiled_Mscatter(slope, zp, Ms, alpha, 
               mag_scatter=0.4, size=100000, seed=0):
    
    np.random.seed(seed)
    
    randMGAG = np.random.uniform(low=-25, high=-13, size=size*10)
    randU = np.random.uniform(low=0, high=1, size=size*10)
    randLfunct = LFfunction(randMGAG, Ms, alpha)
    indx, = np.where(randU<randLfunct)
    simulMag  = randMGAG[indx]
    simulWimx = (simulMag-zp)/slope + 2.5
    
    real_mag  = simulMag[:size]
    simulWimx = simulWimx[:size]
    
    
    ## scattring along the magnitude axis
    N = len(real_mag)
    scatterMAG = np.random.normal(0, mag_scatter, N)
    simulMag = real_mag + scatterMAG
    
    # real mag - scattered mag
    bias = real_mag-simulMag
    
    return bias, simulMag, simulWimx
    
########################################################
def simulFiled_Wscatter(slope, zp, Ms, alpha, 
               mag_scatter=0.4, size=100000, seed=0):
    
    np.random.seed(seed)
    
    randMGAG = np.random.uniform(low=-24.5, high=-13, size=size*10)
    randU = np.random.uniform(low=0, high=1, size=size*10)
    randLfunct = LFfunction(randMGAG, Ms, alpha)
    indx, = np.where(randU<randLfunct)
    simulMag  = randMGAG[indx]
    simulWimx = (simulMag-zp)/slope + 2.5
    
    simulMag = simulMag[:size]
    reallWimx = simulWimx[:size]
    
    
    ## scattring along the magnitude axis
    N = len(simulMag)
    scatterW = np.random.normal(0, np.abs(mag_scatter/slope), N)
    simulWimx = reallWimx + scatterW

    # real mag - scattered mag
    bias = simulMag-(slope*(simulWimx-2.5)+zp)
    
    return bias, simulMag, simulWimx

########################################################

def simulFiled_MWscatter(slope, zp, Ms, alpha, 
               mag_scatter=0.4, size=100000, seed=0):
    
    np.random.seed(seed)
    
    randMGAG = np.random.uniform(low=-24.5, high=-13, size=size*10)
    randU = np.random.uniform(low=0, high=1, size=size*10)
    randLfunct = LFfunction(randMGAG, Ms, alpha)
    indx, = np.where(randU<randLfunct)
    simulMag  = randMGAG[indx]
    simulWimx = (simulMag-zp)/slope + 2.5
    
    real_mag  = simulMag[:size]
    reallWimx = simulWimx[:size]
    
    
    ## scattring along the magnitude axis
    N = len(real_mag)
    
    M = real_mag*0
    W = reallWimx*0
    
    for i in range(N):
    
        eM = np.random.uniform(low=0, high=mag_scatter, size=1)[0]
        eW = np.abs(np.sqrt(mag_scatter**2-eM**2)/slope)
        scatterMAG = np.random.normal(0, eM, 1)[0]
        scatterW = np.random.normal(0, eW, 1)[0]
        
        M[i] = scatterMAG + real_mag[i]
        W[i] = scatterW + reallWimx[i]
    

    # real mag - scattered mag
    bias = real_mag - M 
    
    return bias, M, W, real_mag, reallWimx


########################################################
def simulFiled_ensemble(M0_lst, slope, zp, 
                        bias, simulMag, simulWimx,
                        d_mag=0.05, dW=None):
    
    delta_lst = np.asarray(M0_lst)*0.
    
    for i, M0 in enumerate(M0_lst):
        
        if not dW is None:
            logWimx = (M0-zp)/slope + 2.5
            weights = Normal(simulMag, M0, d_mag)*Normal(simulWimx, logWimx, dW)
        else:
            weights = Normal(simulMag, M0, d_mag)
        
        delta_lst[i] = np.sum(bias*weights)/np.sum(weights) 

    return delta_lst

########################################################
def simulCluster(Mlim_lst, slope, zp, 
                        simulMag, simulWimx, cut_alpha=None):
    
    
    delta_lst = Mlim_lst*0.    
    
    for i, Mlim in enumerate(Mlim_lst):   
        
        if cut_alpha is None:
            indx, = np.where(simulMag<Mlim)
        
        else:
            w = (Mlim-zp)/slope
            cut_beta = Mlim-cut_alpha*w
            Mcut = cut_alpha*(simulWimx-2.5)+cut_beta
            indx, = np.where(simulMag<Mcut)
        
        
        simulMag_obs = simulMag[indx]
        simulWimx_obs = simulWimx[indx]
        simulMag_real =  slope*(simulWimx_obs-2.5)+zp
      
        # real mag - observed mag
        bias = simulMag_real-simulMag_obs 
          
        delta_lst[i] = np.mean(bias)
    

    return delta_lst
        
########################################################
def iterate(SZ, seed=0, 
            mag_scatter=0.56, Mlim = -17, 
            along_MAG=True, Simul_size = 100000,
            Ms=-22, alpha=-1):
########################################################
    slope= SZ[0]; zp=SZ[1]
    
    np.random.seed(seed)
    size = Simul_size

    randMGAG = np.random.uniform(low=-25, high=-13, size=size*10)
    randU = np.random.uniform(low=0, high=1, size=size*10)
    randLfunct = LFfunction(randMGAG, Ms, alpha)
    indx, = np.where(randU<randLfunct)
    simulMag  = randMGAG[indx]
    simulWimx = (simulMag-zp)/slope + 2.5

    
    if along_MAG:
        real_mag = simulMag[:size]
        simulWimx = simulWimx[:size]
        ## scattring along the magnitude axis
        N = len(real_mag)
        scatterMAG = np.random.normal(0, mag_scatter, N)
        simulMag = real_mag + scatterMAG
    
    else:
        simulMag = simulMag[:size]
        reallWimx = simulWimx[:size]
        ## scattring along the magnitude axis
        N = len(simulMag)
        scatterW = np.random.normal(0, np.abs(mag_scatter/slope), N)
        simulWimx = reallWimx + scatterW    
    

    indx, = np.where(simulMag<Mlim)
    simulMag_obs = simulMag[indx]
    simulWimx_obs = simulWimx[indx]


    fit, cov = curve_fit(linfit, simulMag_obs, simulWimx_obs-2.5)
    m, b = fit
    m0 = 1./m
    b0 = -b/m
    
    return [m0, b0]

def Cost(SZ0, SZ):
    
    slope0 = SZ0[0]; zp0 = SZ0[1]
    slope  = SZ[0] ; zp  = SZ[1]
    return np.sqrt(((zp0-zp)/zp0)**2+((slope0-slope)/slope0)**2)








def getSimuSZ(SZ0, thresh=0.0001, along_MAG=True, verbose=False, seed=0, 
            mag_scatter=0.56, Mlim = -17, 
            Simul_size = 100000,
            Ms=-22, alpha=-1):
    
    SZ1 = SZ0
    SZ =  iterate(SZ1, 
                      along_MAG=along_MAG,
                      seed=seed,
                      mag_scatter=mag_scatter, Mlim = Mlim,
                      Simul_size = Simul_size,
                      Ms=Ms, alpha=alpha)
    d = Cost(SZ0, SZ)
    i = 0 
    while d>thresh:

        if verbose:
            print SZ0, SZ1, Cost(SZ0, SZ), SZ
        
        
        i1_ds = iterate([SZ1[0]+0.01, SZ1[1]], 
                      along_MAG=along_MAG,
                      seed=seed,
                      mag_scatter=mag_scatter, Mlim = Mlim,
                      Simul_size = Simul_size,
                      Ms=Ms, alpha=alpha)
        i2_ds = iterate([SZ1[0]-0.01, SZ1[1]], 
                      along_MAG=along_MAG,
                      seed=seed,
                      mag_scatter=mag_scatter, Mlim = Mlim,
                      Simul_size = Simul_size,
                      Ms=Ms, alpha=alpha)
        
        i1_dz = iterate([SZ1[0], SZ1[1]+0.01], 
                      along_MAG=along_MAG,
                      seed=seed,
                      mag_scatter=mag_scatter, Mlim = Mlim,
                      Simul_size = Simul_size,
                      Ms=Ms, alpha=alpha)
        i2_dz = iterate([SZ1[0], SZ1[1]-0.01], 
                      along_MAG=along_MAG,
                      seed=seed,
                      mag_scatter=mag_scatter, Mlim = Mlim,
                      Simul_size = Simul_size,
                      Ms=Ms, alpha=alpha)
        
        
        d_dS =  (Cost(SZ0, i1_ds)-Cost(SZ0, i2_ds))/0.02
        d_dz =  (Cost(SZ0, i1_dz)-Cost(SZ0, i2_dz))/0.02

        SZ1 = [SZ1[0] - 0.1*d_dS, SZ1[1] - 0.1*d_dz]
        SZ =  iterate(SZ1, 
                      along_MAG=along_MAG,
                      seed=seed,
                      mag_scatter=mag_scatter, Mlim = Mlim,
                      Simul_size = Simul_size,
                      Ms=Ms, alpha=alpha)
        
        d = Cost(SZ0, SZ)
    
        i+=1
        if i%10==0:
            print '# of iter:', i, ' Cost: ', '%.4f'%d, SZ 
        
        
    return SZ1
########################################################

























