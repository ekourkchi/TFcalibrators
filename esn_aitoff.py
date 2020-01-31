## Importing Important Python Libraries
import sys
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Polygon, Ellipse
import numpy as np
from math import *
from time import time
import wl_to_rgb as col
import random
from astropy.io import ascii
from astropy.table import Table, Column 
import pyfits
import pylab as py
from kapteyn import wcs
from astropy import coordinates as coord
from astropy import units as unit

import matplotlib.colors as colors
import matplotlib.cm as cmx


# **************************************
# Global Variables
# Physical Constants
# **************************************
H0 = 75.           # hubble constant
sglV = 102.8806    # M87 center - super galactic longitude
sgbV = -2.3479     # M87 center - super galactic latitude
G = 6.67384E-11   # Gravitational constant
M_sun = 1.9891E30  # Solar maas [Kg]
Mpc_km = 3.08567758E19
#t0_age = Mpc_km / H0   # Universe Age [sec]
Yr= 365.25 * 24 * 3600    # [sec]
t0_age = 13.8E9 * Yr # Universe Age [sec] - 13.8 Gyr
h = 0.75  # hubble constant
# **************************************
col_pallet = ['darkviolet', 'blue', 'deepskyblue', 'forestgreen', 'y', 'gold', 'darkorange', 'red', 'magenta', 'maroon', 'sienna', 'slategrey', 'black']
vel_pallet = [0, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 14000, 150000]

#################################################################
def color_table(Vls):
  
  for i in range(len(vel_pallet)-1):
    
    if vel_pallet[i] <= Vls and Vls < vel_pallet[i+1]:
      return col_pallet[i]
  return   colour  


def vel_limits(p, Vls):
  
  if vel_pallet[p] <= Vls and Vls < vel_pallet[p+1]:
    return True
  else: 
    return False


def vel_indx(Vls):
  
  for i in range(len(vel_pallet)-1):
    if vel_pallet[i] <= Vls and Vls < vel_pallet[i+1]:
      return i
  return 0
  
#################################################################
################################################################ 

def plot_border(l0):
  
  
  X = []
  Y = []

  
  l = 0.
  for b in np.arange(-90,90,0.01):
    x, y = xymap(l,b,l0)
    X.append(x)
    Y.append(y)
  l = 360
  for b in np.arange(90,-90,-0.01):
    x, y = xymap(l,b,l0)
    X.append(x)
    Y.append(y)
  
  X.append(X[0])
  Y.append(Y[0])

  plt.plot(X, Y, '-', markersize = 1, linewidth=2., color='black')   # '#0D1891'
  
################################################################ 
################################################################ 

def xymap(l,b,l0):
  
  t = theta(b)
  x = 180.-((2*90.)/180.)*(l0-l)*cos(t)
  y = 90.*sin(t)
  
  return x, y 


#################################################################
################################################################# 
# b is galactic latitude [deg]
# 2*theta + sin(2*theta) = Pi * sin(b)

def theta(b):
  
  if b == 90.: return pi/2.
  if b == -90.: return -pi/2.
  
  b = b*pi/180.
  theta0 = b
  theta1 = theta0-(2.*theta0 + sin(2.*theta0) - pi * sin(b))/(2.+2.*cos(2.*theta0))
  
  while abs(theta1-theta0) > 0.01:
    theta0 = theta1
    theta1 = theta0-(2.*theta0 + sin(2.*theta0) - pi * sin(b))/(2.+2.*cos(2.*theta0))
  
  return theta1
  
################################################################ 
def xymap_aitoff(x,y):
  
  x0 = (180.-x)*pi/180.
  y0 = y*pi/180.
  
  return x0, y0

def esn_aitoff_patch(ax, x0, y0, d, l0, color='blue'):
  
  #vertices = []
  #x, y = xymap(x0,y0, l0) 
  #vertices.append([x,y])
  #x, y = xymap(x0,y0+d, l0) 
  #vertices.append([x,y])
  #x, y = xymap(x0+d,y0+d, l0) 
  #vertices.append([x,y])
  #x, y = xymap(x0+d,y0, l0) 
  #vertices.append([x,y])
  
  
  vertices = []
  x, y = xymap_aitoff(x0,y0) 
  vertices.append([x,y])
  x, y = xymap_aitoff(x0,y0+d) 
  vertices.append([x,y])
  x, y = xymap_aitoff(x0+d,y0+d) 
  vertices.append([x,y])
  x, y = xymap_aitoff(x0+d,y0) 
  vertices.append([x,y])  
  
  
  
  ax.add_patch(Polygon(vertices, closed=True, fill=True, color=color))

##################################################################################################################################
def median_error(X):
  
  size = len(X)
  X    = np.sort(X) 
  mean = np.median(X)
  
  u_err = X[int(round(0.84*size))] - mean
  l_err = mean - X[int(round(0.16*size))]
  
  return mean, u_err, l_err

########################################################################## 
def file_deliver(filee):
  
  try:
    mytable = np.genfromtxt(filee , delimiter='|', filling_values="-100000", names=True, dtype=None )
   
  except:
    print "\n[Error] The catalog \""+filee+"\" is not available, or has a wrong format ..."
    exit(1)
  
  id         = mytable['pgc']
  flag       = mytable['flag']
  sgl        = mytable['sgl']
  sgb        = mytable['sgb']
  gl         = mytable['gl']
  gb         = mytable['gb']  
  ra         = mytable['ra']
  dec        = mytable['dec']
  Ks         = mytable['Ks']
  Vls        = mytable['Vls']
  R_2t2      = mytable['R2t_lum']
  nest       = mytable['nest']
  dcf2       = mytable['dcf2']
  ed         = mytable['ed']
  objname    = mytable['objname']   
  mDist      = mytable['mDist']
  mDistErr   = mytable['mDistErr']
  sigmaP_lum = mytable['sigmaP_lum']
  
  print "\n[Success] The catalog \""+filee+"\" was sucessfully loaded ..."
  print "and has "+str(len(id))+" entries ...\n"
  
  
  out_table = {'pgc':id, 'flag':flag, 'sgl': sgl, 'sgb':sgb, 'gl':gl, 'gb':gb, 'ra':ra, 'dec':dec, 'Ks':Ks, 'Vls':Vls,  \
               'R2t_lum':R_2t2, 'nest':nest, 'dcf2':dcf2, 'ed':ed, 'objname':objname, \
		'mDist':mDist, 'mDistErr':mDistErr, 'sigmaP_lum':sigmaP_lum}
  return out_table
    
  
  
  
  
########################################################################## 
########################################################################## 
########################################################################## 
  
if __name__ == '__main__':
  
  l0 = 180
  #fig = plt.figure(figsize=(12, 12*0.5), dpi=100)
  #ax = fig.add_axes([0.13, 0.13, 0.80,  0.80]) 
  #plt.ylim(-100,100)
  #plt.xlim(380,-20)
  #plt.xlabel("SGL (deg)", fontsize=20)
  #plt.ylabel("SGB (deg)", fontsize=20)
  #plt.yticks(fontsize=16)
  #plt.xticks(fontsize=16)
  #plot_border(l0)
  


  

  deg = 1.0
  a = pyfits.open('EBV.1.deg.fits')
  d = a[1].data
  
  
  SGL  = d['SGL']
  SGB  = d['SGB']
  EBV = d['EBV']
  
  fig = plt.figure(figsize=(12, 12*0.5), dpi=100)
  ax = fig.add_subplot(111, projection="aitoff")
  plt.title("Supergalactic Aitoff Projection of the TF Calibrator Cluster", y=1.08)
  ax.grid(True)
  ax.set_xticklabels([])
  plt.subplots_adjust(top=0.95, bottom=0.0)


  ax.annotate(r'$0^o$', (pi-0.1,pi/3.), size=11, color='black')
  ax.annotate(r'$90^o$', (pi/2.-0.2,pi/3.), size=11, color='black')
  ax.annotate(r'$180^o$', (-0.2,pi/3.), size=11, color='black')
  ax.annotate(r'$270^o$', (-pi/2.-0.2,pi/3.), size=11, color='black')
  

################################################################ 

  #for x0 in np.arange(0,360,deg):
    #for y0 in np.arange(-90,90,deg):
      #esn_aitoff_patch(ax, x0, y0, deg, l0, facecolor='black', alpha=None)
      

################################################################ 
  
  median, u_err, l_err = median_error(EBV)
  
  X = []
  Y = []
  
  
  jet = cm = plt.get_cmap('Greys')
  cNorm  = colors.Normalize(vmin=0, vmax=5)
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
  
  
  for i in range(len(SGB)):
    if EBV[i]>2.0: val = 3
    elif EBV[i]>1.0: val = 2.5
    elif EBV[i]>median+u_err: val= 2.0
    elif EBV[i]<=median+u_err and EBV[i]>median: val= 1.0
    elif EBV[i]>median-l_err and EBV[i]<=median: val= 0.50
    else: val= 0
    
    
     
    
    
    colorVal = scalarMap.to_rgba(val)
    esn_aitoff_patch(ax, SGL[i], SGB[i], deg, l0, color=colorVal)
    
    
    #if EBV[i]>1.0:
      #x, y = xymap(SGL[i], SGB[i], l0)
      #X.append(x)
      #Y.append(y)
      #X.append((180-SGL[i])*pi/180)
      #Y.append(SGB[i]*pi/180.)      
  

  #vertices = [[0,0],[0,1],[1,1],[1,0]]
  
  #ax.plot(X,Y, 'ro', markersize = 3, alpha=0.3)
  #ax.add_patch(Polygon(vertices, closed=True, fill=True, color='blue'))
  
################################################################ 
  #mytable = file_deliver('all.iter.2.v39.group')
  B_North   = np.genfromtxt('EDD_Bren_2MRS_North.csv' , delimiter='|', filling_values=-1, 
                          names=True, dtype=None, encoding=None)
  B_South   = np.genfromtxt('EDD_Bren_2MRS_South.csv' , delimiter='|', filling_values=-1, 
                          names=True, dtype=None, encoding=None)
  
  
  B_North_PGC  = B_North['PGC']
  B_North_Nest = B_North['Nest']
  B_North_Vhel = B_North['Vhel']
  B_North_SGL  = B_North['SGL']
  B_North_SGB  = B_North['SGB']
  B_North_Vmod = B_North['Vbw']

  B_North_PGC  = np.concatenate((B_South['PGC'], B_North_PGC))
  B_North_Nest = np.concatenate((B_South['Nest'], B_North_Nest))
  B_North_Vhel = np.concatenate((B_South['Vhel'], B_North_Vhel))
  B_North_SGL  = np.concatenate((B_South['SGL'], B_North_SGL))
  B_North_SGB  = np.concatenate((B_South['SGB'], B_North_SGB))
  B_North_Vmod = np.concatenate((B_South['Vbw'], B_North_Vmod))
  
  
  
  Clusters = {}
  Clusters[100001] = 'Co'
  Clusters[100002] = 'Vi'
  Clusters[100003] = 'Ce'
  Clusters[100005] = 'A13'
  Clusters[100006] = 'Hy'
  Clusters[100007] = 'He'  
  Clusters[100008] = 'UMa'
  Clusters[100014] = 'An'  
  Clusters[100018] = 'N40'
  Clusters[100030] = 'Ca'
  Clusters[200003] = 'A26'    
  Clusters[200005] = 'N41'  
  Clusters[200006] = 'N50'    
  Clusters[200015] = 'Fo'    
  Clusters[200012] = 'A40'    
  Clusters[200016] = 'A26'  
  Clusters[200017] = 'A53'    
  Clusters[200037] = 'N70'      
  Clusters[200045] = 'N80'    
  Clusters[200092] = 'Pe'  

      
  
  
  
  for key in Clusters:
      
      indx, = np.where(B_North_Nest==key)
      Bpgc= B_North_PGC[indx]
      vh  = B_North_Vhel[indx]
      sgl = B_North_SGL[indx]
      sgb = B_North_SGB[indx]
      vm  = B_North_Vmod[indx]
      
      

      print len(Bpgc), Clusters[key], vm[0]
      
      for i in range(len(Bpgc)):
          x, y = xymap_aitoff(sgl[i], sgb[i])
          l, = ax.plot([x], [y], '.', markersize = 1, color=color_table(vm[0]))
      
      if key==200005:
          x, y = xymap_aitoff(sgl[i], sgb[i]+2)
          ax.text(x,y, Clusters[key])
      else:
          ax.text(x,y, Clusters[key])

  
  for ra in  np.linspace(0,360, 1000):
      
      dec = 0 
      tr = wcs.Transformation("equatorial j2000 j2000", "supergalactic")     
      sgl, sgb = tr((ra, dec))
      x, y = xymap_aitoff(sgl, sgb)
      l, = ax.plot([x], [y], '.', markersize = 1, color='navy')
      
      
      
      
      

    

    #r = 180.*atan(R_2t2[i]/Dist)/pi

    
    #d_theta = 0.01
    #u = np.arange(0,2*pi,d_theta)
    #X = u*0.
    #Y = u*0.
    
    #RA0  = sgl_gal[i]
    #DEC0 = sgb_gal[i]
    
    #for q in range(0,len(u)):
       #a = r*np.cos(u[q]) + RA0
       #b = r*np.sin(u[q]) + DEC0
       #x, y = xymap(a, b, l0)
       #X[q] = x
       #Y[q] = y
       
    #if r <= 20:

        #line, = plt.plot(X,Y, '-', markersize = 2, color=colour, picker=False) 
	#line.set_dashes([8, 3]) 


    #i+=1  
    #X = []
    #Y = []
    #while i<N and flag[i]==1: 

      #x, y = xymap(sgl_gal[i], sgb_gal[i], l0)
      #X.append(x)
      #Y.append(y)
      #i+=1

    #point, = plt.plot(X, Y, 'o', markersize = 1, color=colour, markeredgecolor = colour)

    
    #if i<N and flag[i]==2: 
      #gr = 2
    #else:
       #break  
################################################################ 

  plt.show()









