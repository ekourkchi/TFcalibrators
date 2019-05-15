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


inFile  = 'all.alarms'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_alarm = table['pgc']

inFile  = 'calibrators.alarms'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_calibrators = table['pgc']




for jj in range(len(pgc_ESN)):
    if QA_wise[jj] == 0 :
        if pgc_ESN in pgc:
            print pgc 

sys.exit()


######################################

#monList = []
#for i in range(len(pgc_alarm)):
    #if pgc_alarm[i] in pgc_ESN and not pgc_alarm[i] in monList:
        #indx, = np.where(pgc_ESN==pgc_alarm[i])
        #if Sqlt[indx]>1 and inc[indx]>0:
            #monList.append(pgc_alarm[i])
            #if not pgc_alarm[i] in pgc_calibrators:
                #print pgc_alarm[i]
            
#sys.exit()

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
