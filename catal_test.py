#!/usr/bin/python
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, Column 

### external codes
from redTools import *
from Kcorrect import *
from linear_mcmc import *


########################################################### Begin
ctl   = np.genfromtxt("NEST_100001.csv" , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
PGC_calib = ctl['PGC']

calibLST = []
calibLST += ["NEST_100002.csv"]
calibLST += ["NEST_100003.csv"]
calibLST += ["NEST_100005.csv"]
calibLST += ["NEST_100006.csv"]
calibLST += ["NEST_100007.csv"]
calibLST += ["NEST_100008.csv"]
calibLST += ["NEST_100014.csv"]
calibLST += ["NEST_100018.csv"]
calibLST += ["NEST_100030.csv"]
calibLST += ["NEST_120002.csv"]
calibLST += ["NEST_200003.csv"]
calibLST += ["NEST_200005.csv"]
calibLST += ["NEST_200006.csv"]
calibLST += ["NEST_200012.csv"]
calibLST += ["NEST_200015.csv"]
calibLST += ["NEST_200016.csv"]
calibLST += ["NEST_200017.csv"]
calibLST += ["NEST_200037.csv"]
calibLST += ["NEST_200045.csv"]
calibLST += ["NEST_200092.csv"]

for cluster in calibLST:
    ctl   = np.genfromtxt(cluster , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
    PGC_calib = np.concatenate((PGC_calib, ctl['PGC']))

########################################################### Begin

inFile = 'ESN_HI_catal_calib.csv'
table = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None, encoding=None)
pgc = table['pgc']
Sqlt = table['Sqlt']
Wqlt = table['Wqlt']

#ctl   = np.genfromtxt('NEST_100002.csv' , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
#PGC = ctl['PGC']

#mask = np.isin(pgc, PGC)

#P=pgc[mask]
#S=Sqlt[mask]
#W=Wqlt[mask]

#for i in range(len(P)):
    #print P[i], S[i], W[i]

#sys.exit()



inFile  = '../GBT_helene_candidates.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_helene    = table['pgc']
source_helene    = table['source']
status_helene    = table['status']

inFile  = '../ADHI.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_adhi    = table['PGC']



inFile  = '../ALFALFA100.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_alfalfa = table['PGC']

inFile  = '../Cornel_HI.csv'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_cornel = table['PGC']

inFile  = '../preDigital.csv'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_preDigit = table['PGC']

inFile  = '../EDD_distance_cf4_v24.csv'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_ESN = table['pgc']
RA = table['ra']
DEC = table['dec']
Sqlt = table['Sqlt']
Wqlt = table['Wqlt']
R255_r = table['R255_r']
R255_w1 = table['R255_w1']
Vhel = table['Vhel']
g_mag = table['g_mag']
r_mag = table['r_mag']
i_mag = table['i_mag']
inc   = table['inc']
QA_sdss = table['QA_sdss']
QA_wise = table['QA_wise']
w1_mag  = table['w1_mag']
w2_mag  = table['w2_mag']





#for jj in range(len(pgc_helene)):
    #if not pgc_helene[jj] in pgc_ESN:
        #if status_helene[jj].strip()=='OK' or status_helene[jj].strip()=='CO' or status_helene[jj].strip()=='RE':
            #print pgc_helene[jj], status_helene[jj].strip()

#sys.exit()


######################################
for id in PGC_calib:
    if id in pgc_ESN:
        indx, = np.where(pgc_ESN==id)
        if QA_wise[indx][0]==0:
            print id

            
sys.exit()

######################################


ii=[];jj=[]
for i in range(len(pgc_ESN)):
    id = pgc_ESN[i]
    if not id in pgc_adhi:
        if not id in pgc_alfalfa: 
            if not id in pgc_cornel:
                if id in pgc_preDigit:
                    ii.append(i)
                    print i, pgc_ESN[i]
                else:
                    jj.append(i)


ii = np.asarray(ii)
jj = np.asarray(jj)

indx = ii
ID    = pgc_ESN[indx]

print len(indx), len(ID)
print ID[0:10]


Vhel  = Vhel[indx]
RA    = RA[indx]
DEC   = DEC[indx]
g_mag = g_mag[indx]
r_mag = r_mag[indx]
i_mag = i_mag[indx]
Sqlt  = Sqlt[indx]
Wqlt  = Wqlt[indx]
INC = inc[indx]


R255_r  = R255_r[indx]
R255_w1 = R255_w1[indx]

R255 = R255_r  * 0.



################################
inFile  = '../All_LEDA_EDD.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

pgc_leda = table['pgc']
objname_leda = table['objname']
it = table['it']
bt =  table['bt']
Name = []
Bt = []
It = []
for i in range(len(ID)):
    if ID[i] in pgc_leda:
        indx, = np.where(pgc_leda==ID[i])
        Name.append(objname_leda[indx])
        
        Bmag = it[indx]
        Imag = bt[indx]
        if Bmag<0:  Bmag=0
        if Imag<0:  Imag=0
        Bt.append(Bmag)
        It.append(Imag)
        
        if Sqlt[i]>=2:
            R255[i]=R255_r[i]
        elif Wqlt[i]>=2:
            R255[i]=R255_w1[i]
        elif Sqlt[i]==1 and Wqlt[i]==0:
            R255[i]=R255_r[i]
        elif Sqlt[i]==0 and Wqlt[i]==1:
            R255[i]=R255_w1[i]
            
    else:
        Name.append('')
        Bt.append(0)
        It.append(0)        
        
    


Name = np.asarray(Name)
Bt = np.asarray(Bt)
It = np.asarray(It)


indx = []
for i in range(len(ID)):
    if INC[i]!=0 and (Sqlt[i]>3 or Wqlt[i]>3) and DEC[i]>-45 and Bt[i]<15 and It[i]<16 and R255[i]<9 and INC[i]>60 and (Bt[i]>0 or It[i]>0):
        indx.append(i)

indx = np.asarray(indx)
        
    

ID    = ID[indx]
Vhel  = Vhel[indx]
RA    = RA[indx]
DEC   = DEC[indx]
g_mag = g_mag[indx]
r_mag = r_mag[indx]
i_mag = i_mag[indx]
Sqlt  = Sqlt[indx]
Wqlt  = Wqlt[indx]
INC   = INC[indx]
R255  = R255[indx]
Name  = Name[indx]
Bt    = Bt[indx]
It    = It[indx]
################################
    
myTable = Table()

myTable.add_column(Column(data=ID, name='pgc'))
myTable.add_column(Column(data=Name, name='Name'))
myTable.add_column(Column(data=RA, name='RA', format='%0.4f'))
myTable.add_column(Column(data=DEC, name='DEC', format='%0.4f'))
myTable.add_column(Column(data=Vhel, name='Vhel', format='%d'))
myTable.add_column(Column(data=Bt, name='B', format='%0.2f'))
myTable.add_column(Column(data=It, name='I', format='%0.2f'))
myTable.add_column(Column(data=R255, name='R255', format='%0.2f'))
myTable.add_column(Column(data=INC, name='inc'))
myTable.add_column(Column(data=Sqlt, name='Sqlt', dtype=np.dtype(int)))
myTable.add_column(Column(data=Wqlt, name='Wqlt', dtype=np.dtype(int)))


myTable.write('tmp.csv', format='ascii.fixed_width',delimiter=',', bookend=False, overwrite=True) 


#myTable.write('NOTinPreDigital.csv', format='ascii.fixed_width',delimiter=',', bookend=False, overwrite=True) 

#myTable.write('inPreDigital.csv', format='ascii.fixed_width',delimiter=',', bookend=False, overwrite=True) 
