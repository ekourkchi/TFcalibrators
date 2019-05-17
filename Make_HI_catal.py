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
inFile  = '../preDigital.csv'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
pgc_preDigit = table['PGC']
W20 = table['W20']
e_W20 =  table['e_W20']
Flux_HI_preDigit =  (10**table['logFHI'])/0.236
W_m50_preDigit = 1.012*W20-17.9
e_W_m50_preDigit = 1.012*e_W20



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

inFile  = '../ADHI.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)


pgc    = table['PGC']
Vh_av  = table['Vh_av']
Wmx_av = table['Wmx_av']
eW_av  = table['eW_av']
N_av   = table['N_av']
Wmx1   = table['Wmx1']
e_W1   = table['e_W1']
SN1    = table['SN1']
Flux1  = table['Flux1']
Wmx2   = table['Wmx2']
e_W2   = table['e_W2']
SN2    = table['SN2']
Flux2  = table['Flux2']
Wmx3   = table['Wmx3']
e_W3   = table['e_W3']
SN3    = table['SN3']
Flux3  = table['Flux3']

N = len(pgc)

Wmx  = np.zeros(N)
eWmx = np.zeros(N)
F_av  = np.zeros(N)
NN_av = np.zeros(N)

for i in range(N):
    
    n = 0
    W_tot = 0
    eWmx_tot = 0
    #F_tot = 0
    if e_W1[i]<= 20 and e_W1[i]>=0:
        n+=1
        W_tot+=1.*Wmx1[i]/e_W1[i]**2
        #F_tot+=Flux1[i]
        eWmx_tot+=1./e_W1[i]**2
    if e_W2[i]<= 20 and e_W2[i]>=0:
        n+=1
        W_tot+=1.*Wmx2[i]/e_W2[i]**2
        #F_tot+=Flux2[i]
        eWmx_tot+=1./e_W2[i]**2
    if e_W3[i]<= 20 and e_W3[i]>=0:
        n+=1
        W_tot+=1.*Wmx3[i]/e_W3[i]**2
        #F_tot+=Flux3[i]  
        eWmx_tot+=1./e_W3[i]**2

    if n>0:
        Wmx[i]  = W_tot/eWmx_tot
        #F_av[i]  = F_tot/n
        eWmx[i] = math.sqrt(1./eWmx_tot)
        NN_av[i] = n
        
    n=0
    F_tot = 0
    if Flux1[i]>0:
        n+=1
        F_tot+=Flux1[i]
    if Flux2[i]>0:
        n+=1
        F_tot+=Flux2[i]
    if Flux3[i]>0:
        n+=1
        F_tot+=Flux3[i]
    if n>0: F_av[i]  = 1.*F_tot/n
        
#x = []
#y = []
#for i in range(N):
    #if pgc[i] in pgc_preDigit:
        
        #indx, = np.where(pgc_preDigit==pgc[i])
        #print Wmx[i], W_m50_preDigit[indx]
        #if F_av[i]>0 and Flux_HI_preDigit[indx][0]>0:
            #x.append(F_av[i])
            #y.append(Flux_HI_preDigit[indx][0])
        

#plt.plot(x,y,'k.')
#plt.plot([0,800],[0,800],'r--')
#plt.show()

#sys.exit()

########################################################### Begin
inFile  = '../ALFALFA100.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

pgc_alfalfa = table['PGC']
Vhel_alfalfa = table['Vhel']
W_M50_alfalfa = table['W_M50']
e_W_alfalfa = table['e_W']
F_alfalfa = table['F']
e_F_alfalfa = table['e_F']
SNR_alfalfa = table['SNR']

N_alfalfa = len(pgc_alfalfa)

Wm6 = W_M50_alfalfa - 6
e_Wm6 = e_W_alfalfa

#for i in range(N_alfalfa):
    
    #if pgc_alfalfa[i] in pgc:
        
        #i_ind = np.where(pgc==pgc_alfalfa[i])
        
        #if Wmx_av[i_ind][0]>0:
            ###print pgc[i_ind][0], Wmx_av[i_ind][0], Wmx[i_ind][0], pgc_alfalfa[i], Wm6[i]
            #plt.plot(Wmx_av[i_ind][0], Wm6[i], 'r.')
            ###plt.plot(F_av[i_ind][0], F_alfalfa[i], 'g.')
#plt.show()

########################################################### Begin
inFile  = '../Cornel_HI.csv'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None)

pgc_cornel = table['PGC']
Fc_cornel = table['Fc']
SNR_cornel = table['SNR']
Vhel_cornel = table['Vhel']
Wc_cornel = table['Wc']
e_Wc_cornel = table['e_Wc']

Wcfix = 1.015*Wc_cornel-11.0
e_Wcfix = 1.015*e_Wc_cornel

########################################################### Begin
inFile  = '../EDD_distance_cf4_v24.csv'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None)

pgc_ESN = table['pgc']


u_mag = table['u_mag']
g_mag = table['g_mag']
r_mag = table['r_mag']
i_mag = table['i_mag']
z_mag = table['z_mag']

m255_u  = table['m255_u']
m255_g  = table['m255_g']
m255_r  = table['m255_r']
m255_i  = table['m255_i']
m255_z  = table['m255_z']

d_m_ext_u   = table['d_m_ext_u']
d_m_ext_g   = table['d_m_ext_g']
d_m_ext_r   = table['d_m_ext_r']
d_m_ext_i   = table['d_m_ext_i']
d_m_ext_z   = table['d_m_ext_z']

u_mag_prime = m255_u + d_m_ext_u
g_mag_prime = m255_g + d_m_ext_g
r_mag_prime = m255_r + d_m_ext_r
i_mag_prime = m255_i + d_m_ext_i
z_mag_prime = m255_z + d_m_ext_z

w1_mag = table['w1_mag']
w2_mag = table['w2_mag']

m255_w1  = table['m255_w1']
d_m_ext_w1  = table['d_m_ext_w1']

m255_w2  = table['m255_w2']
d_m_ext_w2  = table['d_m_ext_w2']

w1_mag_prime = m255_w1 + d_m_ext_w1
w2_mag_prime = m255_w2 + d_m_ext_w2

ebv = table['ebv']
Vhel = table['Vhel']
Vls = table['Vls']

Sqlt = table['Sqlt']
Wqlt = table['Wqlt']
inc = table['inc']
inc_e = table['inc_e']
inc_n = table['inc_n']
inc_flg = table['inc_flg']
face_on  = table['fon']
inc_note = table['inc_note']

R50_u = table['R50_u']
R50_g = table['R50_g']
R50_r = table['R50_r']
R50_i = table['R50_i']
R50_z = table['R50_z']
R50_w1 = table['R50_w1']
R50_w2 = table['R50_w2']

C82_u = table['C82_u']
C82_g = table['C82_g']
C82_r = table['C82_r']
C82_i = table['C82_i']
C82_z = table['C82_z']
C82_w1 = table['C82_w1']
C82_w2 = table['C82_w2']

ty  = table['ty']
Sba = table['Sba']
Wba = table['Wba']

M = len(pgc_ESN)

Wmx_ESN  = np.zeros(M)
eWmx_ESN = np.zeros(M)
F_av_ESN = np.zeros(M)
eF_av_ESN = np.zeros(M)
flag = np.zeros(M)

alf = 0
adhi = 0 
cornel=0
both = 0 
alll=0

print len(PGC_calib)
print M
iter = 0
jj= 0 
#for id in PGC_calib:
    #if not id in pgc_ESN:
        #print "PGC: ", id

#for id in PGC_calib:
    #if not id in pgc and not id in pgc_alfalfa and not id in pgc_cornel:
        #print "PGC (No HI): ", id

for i in range(M):
    
    flag[i] = -1
    fon = " ".join(face_on[i].split())
    #if (Sqlt[i]>0 or Wqlt[i]>0 or Wqlt[i]<0) and inc_flg[i]==0:
        #flag[i] = 0
    #elif (fon == 'F' or (inc_flg[i]==1 and 'face_on' in inc_note[i])) and Sqlt[i]>4 and Wqlt[i]>4:   
        #flag[i] = 1
    #elif (fon == 'F' or (inc_flg[i]==1 and 'face_on' in inc_note[i])) and Sqlt[i]>3 and Wqlt[i]>3:   
        #flag[i] = 2        
    #elif (fon == 'F' or (inc_flg[i]==1 and 'face_on' in inc_note[i])) and Sqlt[i]>2 and Wqlt[i]>2:   
        #flag[i] = 3           
    
    if True: # inc_flg[i]==0:
        if Sqlt[i]>0:
            flag[i] = 0
        elif r_mag[i]>0:
            flag[i] = 0
        elif Wqlt[i]>0 or Wqlt[i]<0:
            flag[i] = 0
        elif w1_mag[i]>0:
            flag[i] = 0

    if flag[i]>=0:
        iter+=1
    
    n = 0
    W_tot = 0
    eWmx_tot = 0
    
    bool1=False;bool2=False;bool3=False
       
    
    if pgc_ESN[i] in pgc:
        ind = np.where(pgc==pgc_ESN[i])
        if eWmx[ind][0]>0 and eWmx[ind][0]<=20:
            n+=1
            W_tot+=1.*Wmx[ind][0]/eWmx[ind][0]**2
            #F_tot+=F_av[ind][0]
            eWmx_tot+=1./eWmx[ind][0]**2
            if flag[i]>=0: 
                adhi+=1
                bool1=True
        
    
    if pgc_ESN[i] in pgc_alfalfa:
        ind = np.where(pgc_alfalfa==pgc_ESN[i])
        if SNR_alfalfa[ind][0]>=10:    
            n+=1
            W_tot+=1.*Wm6[ind][0]/e_Wm6[ind][0]**2
            #F_tot+=F_alfalfa[ind][0]
            eWmx_tot+=1./e_Wm6[ind][0]**2 
            if flag[i]>=0: 
                alf+=1
                bool2=True
            
    
    
    if n>0:
        Wmx_ESN[i]  = W_tot/eWmx_tot
        eWmx_ESN[i] = math.sqrt(1./eWmx_tot)
    else:
      if pgc_ESN[i] in pgc_cornel:  
          ind = np.where(pgc_cornel==pgc_ESN[i])
          Wmx_ESN[i]  = Wcfix[ind][0]
          eWmx_ESN[i] = e_Wcfix[ind][0]
          if flag[i]>=0:  
              bool3=True
              cornel+=1
    
    if bool1 or bool2: both+=1
    if bool1 or bool2 or bool3: alll+=1
    if pgc_ESN[i] in pgc or pgc_ESN[i] in pgc_alfalfa or pgc_ESN[i] in pgc_cornel:
        jj+=1
    #else:
        #print pgc_ESN[i] 
          
    n = 0
    F_tot = 0 
    bol = False
    if pgc_ESN[i] in pgc_alfalfa:
        ind = np.where(pgc_alfalfa==pgc_ESN[i])
        if F_alfalfa[ind][0]>0:
            n+=1
            F_tot+=F_alfalfa[ind][0]
            eF_av_ESN[i] = e_F_alfalfa[ind][0]
    if pgc_ESN[i] in pgc_cornel:  
        ind = np.where(pgc_cornel==pgc_ESN[i])
        if Fc_cornel[ind][0]>0:
            n+=1
            F_tot+= Fc_cornel[ind][0]
            bol = True
            
    if n>0: 
        F_tot = 1.*F_tot/n
        n = 1
    if pgc_ESN[i] in pgc:
        ind = np.where(pgc==pgc_ESN[i])
        if F_av[ind][0]>0:
            if n==0:
                eF_av_ESN[i] = 0.17*F_av[ind][0]
            else: bol=True
            n+=1
            F_tot+=F_av[ind][0]
    if n>0: 
        F_tot = 1.*F_tot/n
    F_av_ESN[i] = F_tot
    if bol: eF_av_ESN[i] = 0.07*F_av_ESN[i]
    
    ### Sue PreDigital catalog if galaxy is not in any modern digital catalog
    if pgc_ESN[i] in pgc_preDigit and not pgc_ESN[i] in pgc and not pgc_ESN[i] in pgc_alfalfa and not pgc_ESN[i] in pgc_cornel:
          ind, = np.where(pgc_preDigit==pgc_ESN[i])
          
          if W_m50_preDigit[ind][0]>0:
              Wmx_ESN[i]  = W_m50_preDigit[ind][0]
              eWmx_ESN[i] = e_W_m50_preDigit[ind][0]
              F_av_ESN[i] = Flux_HI_preDigit[ind][0]
              eF_av_ESN[i] = 0.2*F_av_ESN[i]
          
          
print 'iter: ', iter 
print 'jj: ', jj
#plt.show() 

m21 = np.zeros(M)
m21_e = np.zeros(M)
logWimx = np.zeros(M)
logWimx_e = np.zeros(M)

for i in range(M):
    if F_av_ESN[i]>0:
        m21[i] = -2.5*np.log10(F_av_ESN[i])+17.40
        m21_e[i] = (2.5/np.log(10))*(eF_av_ESN[i]/F_av_ESN[i])
    else: m21[i] = -1000
    if Wmx_ESN[i]>0:
        deg2rad = 3.1415/180.
        if inc[i]==0:
            alfa = 40.*deg2rad
            d_alfa = 15.*deg2rad
        else: 
            alfa = inc[i]*deg2rad
            d_alfa = inc_e[i]*deg2rad
        logWimx[i] = np.log10(Wmx_ESN[i]/np.sin(alfa))
        logWimx_e[i] = np.sqrt((eWmx_ESN[i]/Wmx_ESN[i])**2+(d_alfa/np.tan(alfa))**2)/np.log(10)
    else: flag[i]=-1
    

          
        
    
index = np.where(flag>=0)

print 'iter2: ', len(index[0]) 

pgc_ESN = pgc_ESN[index]
Wmx_ESN = Wmx_ESN[index]
F_av_ESN = F_av_ESN[index]
eWmx_ESN = eWmx_ESN[index]
eF_av_ESN = eF_av_ESN[index]

inc = inc[index]
inc_e = inc_e[index]
inc_n = inc_n[index]

u_mag = u_mag[index]
g_mag = g_mag[index]
r_mag = r_mag[index]
i_mag = i_mag[index]
z_mag = z_mag[index]

w1_mag = w1_mag[index]
w2_mag = w2_mag[index]

u_mag_prime = u_mag_prime[index]
g_mag_prime = g_mag_prime[index]
r_mag_prime = r_mag_prime[index]
i_mag_prime = i_mag_prime[index]
z_mag_prime = z_mag_prime[index]

w1_mag_prime = w1_mag_prime[index]
w2_mag_prime = w2_mag_prime[index]


ebv = ebv[index]
Vhel = Vhel[index]
Vls = Vls[index]

logWimx = logWimx[index]
m21 = m21[index]
logWimx_e = logWimx_e[index]
m21_e = m21_e[index]

flag = flag[index]
Sqlt = Sqlt[index]
Wqlt = Wqlt[index]

R50_u  = R50_u[index]
R50_g  = R50_g[index]
R50_r  = R50_r[index]
R50_i  = R50_i[index]
R50_z  = R50_z[index]
R50_w1 = R50_w1[index]
R50_w2 = R50_w2[index]

C82_u  = C82_u[index]
C82_g  = C82_g[index]
C82_r  = C82_r[index]
C82_i  = C82_i[index]
C82_z  = C82_z[index]
C82_w1 = C82_w1[index]
C82_w2 = C82_w2[index]

ty  = ty[index]
Sba = Sba[index]
Wba = Wba[index]

################################
inFile  = '../All_LEDA_EDD.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

pgc_leda = table['pgc']
objname_leda = table['objname']
Name = []

for i in range(len(pgc_ESN)):
    indx = np.where(pgc_leda==pgc_ESN[i])
    Name.append(objname_leda[indx][0])
    
Name = np.asarray(Name)
################################
u_mag = u_mag-0.04
z_mag = z_mag+0.02

myTable = Table()

myTable.add_column(Column(data=pgc_ESN, name='pgc'))
myTable.add_column(Column(data=Name, name='Name'))

myTable.add_column(Column(data=Vhel, name='Vhel', format='%0.1f'))
myTable.add_column(Column(data=Vls, name='Vls', format='%0.1f'))

myTable.add_column(Column(data=Wmx_ESN, name='Wmx', format='%0.3f'))
myTable.add_column(Column(data=eWmx_ESN, name='eWmx', format='%0.3f'))

myTable.add_column(Column(data=logWimx, name='logWimx', format='%0.3f'))
myTable.add_column(Column(data=logWimx_e, name='logWimx_e', format='%0.3f'))

myTable.add_column(Column(data=F_av_ESN, name='F21', format='%0.3f'))
myTable.add_column(Column(data=eF_av_ESN, name='eF21', format='%0.3f'))

myTable.add_column(Column(data=m21, name='m21', format='%0.3f'))
myTable.add_column(Column(data=m21_e, name='m21_e', format='%0.3f'))

myTable.add_column(Column(data=u_mag, name='u', format='%0.2f'))
myTable.add_column(Column(data=g_mag, name='g', format='%0.2f'))
myTable.add_column(Column(data=r_mag, name='r', format='%0.2f'))
myTable.add_column(Column(data=i_mag, name='i', format='%0.2f'))
myTable.add_column(Column(data=z_mag, name='z', format='%0.2f'))
myTable.add_column(Column(data=w1_mag, name='w1', format='%0.2f'))
myTable.add_column(Column(data=w2_mag, name='w2', format='%0.2f'))

myTable.add_column(Column(data=u_mag_prime, name='u_', format='%0.2f'))
myTable.add_column(Column(data=g_mag_prime, name='g_', format='%0.2f'))
myTable.add_column(Column(data=r_mag_prime, name='r_', format='%0.2f'))
myTable.add_column(Column(data=i_mag_prime, name='i_', format='%0.2f'))
myTable.add_column(Column(data=z_mag_prime, name='z_', format='%0.2f'))
myTable.add_column(Column(data=w1_mag_prime, name='w1_', format='%0.2f'))
myTable.add_column(Column(data=w2_mag_prime, name='w2_', format='%0.2f'))

myTable.add_column(Column(data=R50_u, name='R50_u', format='%0.2f'))
myTable.add_column(Column(data=R50_g, name='R50_g', format='%0.2f'))
myTable.add_column(Column(data=R50_r, name='R50_r', format='%0.2f'))
myTable.add_column(Column(data=R50_i, name='R50_i', format='%0.2f'))
myTable.add_column(Column(data=R50_z, name='R50_z', format='%0.2f'))
myTable.add_column(Column(data=R50_w1, name='R50_w1', format='%0.2f'))
myTable.add_column(Column(data=R50_w2, name='R50_w2', format='%0.2f'))

myTable.add_column(Column(data=C82_u, name='C82_u', format='%0.2f'))
myTable.add_column(Column(data=C82_g, name='C82_g', format='%0.2f'))
myTable.add_column(Column(data=C82_r, name='C82_r', format='%0.2f'))
myTable.add_column(Column(data=C82_i, name='C82_i', format='%0.2f'))
myTable.add_column(Column(data=C82_z, name='C82_z', format='%0.2f'))
myTable.add_column(Column(data=C82_w1, name='C82_w1', format='%0.2f'))
myTable.add_column(Column(data=C82_w2, name='C82_w2', format='%0.2f'))

myTable.add_column(Column(data=Sba, name='Sba', format='%0.2f'))
myTable.add_column(Column(data=Wba, name='Wba', format='%0.2f'))
myTable.add_column(Column(data=ty, name='Ty', format='%0.1f'))

myTable.add_column(Column(data=ebv, name='ebv'))

myTable.add_column(Column(data=inc, name='inc'))
myTable.add_column(Column(data=inc_e, name='inc_e'))
myTable.add_column(Column(data=inc_n, name='inc_n'))

myTable.add_column(Column(data=flag, name='flag', dtype=np.dtype(int)))
myTable.add_column(Column(data=Sqlt, name='Sqlt', dtype=np.dtype(int)))
myTable.add_column(Column(data=Wqlt, name='Wqlt', dtype=np.dtype(int)))
myTable.write('ESN_HI_catal_all.csv', format='ascii.fixed_width',delimiter=',', bookend=False, overwrite=True) 



