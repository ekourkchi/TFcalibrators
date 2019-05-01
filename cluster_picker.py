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

Coma = np.genfromtxt('NEST_100001_mags_linewidth.csv' , delimiter=',', filling_values=-1, names=True, dtype=None, encoding=None)

pgc_ = Coma['pgc']
logWimx_ = Coma['logWimx']
logWimx_e_ = Coma['logWimx_e']
mag_ = Coma['w1']
Sqlt_ = Coma['Sqlt']
Wqlt_ = Coma['Wqlt']
mag_e_ = mag_*0+0.05

fig = py.figure(figsize=(7, 5), dpi=100)
fig.subplots_adjust(hspace=0.15, top=0.95, bottom=0.15, left=0.15, right=0.95)
ax = fig.add_subplot(111)

ax.plot(logWimx_, mag_, 'k.', picker=10)

#for i in range(len(pgc_)):
        #if Sqlt_[i]>2 and Wqlt_[i]>2:
            #ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], fmt='o', color='orange', markersize=3)
        #else:
            #ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], fmt='o', color='k', markersize=3, markerfacecolor='white')


indx, = np.where(Sqlt_>2)
logWimx_= logWimx_[indx]
logWimx_e_= logWimx_e_[indx]
mag_= mag_[indx]
mag_e_= mag_e_[indx]
Sqlt_= Sqlt_[indx]
Wqlt_= Wqlt_[indx]


indx, = np.where(Wqlt_>2)
logWimx_= logWimx_[indx]
logWimx_e_= logWimx_e_[indx]
mag_= mag_[indx]
mag_e_= mag_e_[indx]
Sqlt_= Sqlt_[indx]
Wqlt_= Wqlt_[indx]            
            
            
x_ = np.linspace(1,4,50)
M,B,samples=linMC(logWimx_, mag_, logWimx_e_, mag_e_)
m = M[0] ; me=0.5*(M[1]+M[2])
b = B[0] ; be=0.5*(B[1]+B[2])
y_, yu, yl = linSimul(samples, x_, size=500)
ax.fill_between(x_, y_+5*yu, y_-5*yl, color='k', alpha=0.2, edgecolor="none")
ax.plot(x_, m*x_+b, 'k--')     
ax.text(1.9,12, "m= "+"%.2f" % m+'$\pm$'+"%.2f" % me, fontsize=12)
ax.text(1.9,13, "b= "+"%.2f" % b+'$\pm$'+"%.2f" % be, fontsize=12)



xLim = (1.85,2.9)
yLim = (19.5,10.5)
ax.set_xlim(xLim)        
ax.set_ylim(yLim) 
ax.set_ylabel(r'$'+'i'+'^*$', fontsize=18)
ax.set_xlabel(r'$Log( W_{mx}^i)$', fontsize=18)

        
def onpick(event):
    ind = event.ind
    print 'pgc', pgc_[ind]
    
    
fig.canvas.mpl_connect('pick_event', onpick)
plt.show()    
