import sys
import os
import os.path
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
import pylab as py
from astropy.table import Table, Column 
import time
import datetime
from bokeh.plotting import *
from bokeh.embed import components
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Range1d, Label, TapTool, OpenURL, CustomJS, CrosshairTool
from scipy.stats import linregress
from scipy import interpolate
from scipy import polyval, polyfit
from scipy import odr
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from linear_mcmc import *
from redTools import *
from Kcorrect import *
########################################################

def getReddening(band1='r', band2='w2'):
    
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
    
    return Aj, Aj_e2, gamma, d_gamma
########################################################
band2 = 'w2'
band2 = 'w2'
inFile = '../reddening/ESN_HI_catal.csv'
scaler, pca = transform(inFile, band1 = 'r', band2 = 'w2')
u = scaler.mean_
s = scaler.scale_
v = scaler.var_
## z = (x-u)/s
##u: mean  s:scale  var=s**2
A = pca.explained_variance_ratio_ 
pca_inv_data = pca.inverse_transform(np.eye(3)) 
p0 = pca_inv_data[0,0]
p1 = pca_inv_data[0,1]
p2 = pca_inv_data[0,2]
########################################################


inFile = 'ESN_HI_catal_calib.csv'
table = getTable(inFile, band1 = 'r', band2 = band2, minWISEqual=2, minSDSSqual=2, clean=False) 
pgc = table['pgc']
logWimx = table['logWimx']
logWimx_e = table['logWimx_e']
inc = table['inc']
r_w1 = table['r_w1']
c21w = table['c21w'] 
Er_w1 = table['Er_w1']
Ec21w = table['Ec21w']
C82  = table['C82_w2']   # concentration 80%/20%
mu50 = table['mu50']    
data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
order_of_keys = ['$Log( W_{mx}^i)$', '$c21W2$', '$\mu 50$']
list_of_tuples = [(key, data[key]) for key in order_of_keys]
data = OrderedDict(list_of_tuples)
n_comp = len(data)
d = pd.DataFrame.from_dict(data)
z_data = scaler.transform(d)
pca_data = pca.transform(z_data)
pc0 = pca_data[:,0]

logWimx_e = table['logWimx_e']
Ec21w = table['Ec21w']
Emu50 = table['Emu50']
table['Epc0'] = np.sqrt((p0*logWimx_e/s[0])**2+(p1*Ec21w/s[1])**2+(p2*Emu50/s[2])**2)    
table['pc0'] = pc0

Epc0  = table['Epc0']
Einc  = table['inc_e']


Au, Au_e, gamma_u, d_gamma_u = getReddening(band1='u')
Ag, Ag_e, gamma_g, d_gamma_g = getReddening(band1='g')
Ar, Ar_e, gamma_r, d_gamma_r = getReddening(band1='r')
Ai, Ai_e, gamma_i, d_gamma_i = getReddening(band1='i')
Az, Az_e, gamma_z, d_gamma_z = getReddening(band1='z')
Aw1, Aw1_e, gamma_w1, d_gamma_w1 = getReddening(band1='w1')
Aw2, Aw2_e, gamma_w2, d_gamma_w2 = getReddening(band1='w2')

Redden = {}
eRed   = {}
Redden['u'] = Au
Redden['g'] = Ag
Redden['r'] = Ar
Redden['i'] = Ai
Redden['z'] = Az
Redden['w1'] = Aw1
Redden['w2'] = Aw1*0.
eRed['u'] = Au_e
eRed['g'] = Ag_e
eRed['r'] = Ar_e
eRed['i'] = Ai_e
eRed['z'] = Az_e
eRed['w1'] = Aw1_e  
eRed['w2'] = Aw1_e*0.

band_lst = ['u', 'g','r','i','z','w1','w2']
dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple", 'w2': "brown"}
########################################################
def plot_array(clusterName='Coma Cluster', nest='Nest_100001'):
    
    cluster = nest+'.csv'
    figname = nest+'.png'
    
    ctl   = np.genfromtxt(cluster , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
    PGC = ctl['PGC']
    
    mask = np.isin(pgc, PGC)
    Wmxi_range = logWimx[mask]
    u_range = table['u'][mask]
    g_range = table['g'][mask]
    r_range = table['r'][mask]
    i_range = table['i'][mask]
    z_range = table['z'][mask]
    w1_range = table['w1'][mask]
    w2_range = table['w2'][mask]
    xLim_0 = min(Wmxi_range)-0.3
    xLim_1 = max(Wmxi_range)+0.2
    mags_range = np.concatenate((u_range,g_range,r_range,i_range,z_range,w1_range,w2_range))
    mags_range = mags_range[np.where(mags_range>0)]
    yLim_0 = max(mags_range)+0.7
    yLim_1 = min(mags_range)-0.7
    xLim   = (xLim_0, xLim_1)
    yLim   = (yLim_0, yLim_1)
    
    fig = py.figure(figsize=(5, 18), dpi=100) 
    fig.subplots_adjust(wspace=0, top=0.98, bottom=0.04, left=0.17, right=0.98)
    gs = gridspec.GridSpec(7, 1) ; p=0
  
    print clusterName, nest
    for band in band_lst:
        
        xlabel = False; ylabel=True
        if band=='w2': xlabel=True
        
        ax = plt.subplot(gs[p]) ; p+=1
        plot_band(ax, PGC, xlabel=xlabel, ylabel=ylabel, band=band, xLim=xLim, yLim=yLim)
        yticks = ax.yaxis.get_major_ticks()
        if band!='w2': yticks[-1].label1.set_visible(False)
        if band!='w2': plt.setp(ax.get_xticklabels(), visible=False)   
        if band=='u': ax.set_title(nest+' ... '+clusterName, fontsize=14)
        
    
    plt.subplots_adjust(hspace=.0, wspace=0)
    fig.savefig(figname)
########################################################
def plot_band(ax, PGC, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band='r', xLim = (1.85,2.9), yLim = (19.5,10.5)):  
    
    A = Redden[band]
    eA = eRed[band]
    rang=dye[band]   
    mag = table[band] - A
    mag_e = np.sqrt((mag*0.+0.05)**2+eA**2)
    color=dye[band]
    pgc_ = []
    logWimx_ = []
    logWimx_e_ = []
    mag_ = []
    mag_e_ = []
    Sqlt_ = []
    Wqlt_ = []
    
    for i in range(len(pgc)):
        id=pgc[i]
        if id in PGC:
            pgc_.append(id)
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
    
    
    for i in range(len(pgc_)):
            if Sqlt_[i]>2 and Wqlt_[i]>2:
                ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], yerr=mag_e_[i], fmt='o', color=color, markersize=3)
            else:
                ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], yerr=mag_e_[i], fmt='o', color='k', markersize=3, markerfacecolor='white')

    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.5, direction='in')
    ax.minorticks_on()
    ax.set_xlim(xLim)        
    ax.set_ylim(yLim)
    
    indx, = np.where(Sqlt_>2)
    pgc_ = pgc_[indx]
    logWimx_= logWimx_[indx]
    logWimx_e_= logWimx_e_[indx]
    mag_= mag_[indx]
    mag_e_= mag_e_[indx]
    Sqlt_= Sqlt_[indx]
    Wqlt_= Wqlt_[indx]
    

    indx, = np.where(Wqlt_>2)
    pgc_ = pgc_[indx]
    logWimx_= logWimx_[indx]
    logWimx_e_= logWimx_e_[indx]
    mag_= mag_[indx]
    mag_e_= mag_e_[indx]
    Sqlt_= Sqlt_[indx]
    Wqlt_= Wqlt_[indx]
    

#### Linear MCMC fit, considering all errors in both directions
    y_ = np.linspace(5,20,50)
    M,B,samples=linMC1D(mag_, logWimx_-2.5, logWimx_e_, nsteps=1000, ignore=200)
    m = M[0] ; me=0.5*(M[1]+M[2])
    b = B[0] ; be=0.5*(B[1]+B[2])
    m0 = 1./m
    b0 = -b/m
    dm0 = np.abs(me/m**2)
    db0 = np.sqrt((me*b)**2+(be*m)**2)/m**2
    x_, xu, xl = linSimul(samples, y_, size=500)
    ax.fill_betweenx(y_, x_+5*xu+2.50, x_-5*xl+2.50, color='k', alpha=0.2, edgecolor="none")
    ax.plot(m*y_+b+2.5, y_, 'k--') 
    Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
    x0 = 0.9*Xlm[0]+0.1*Xlm[1]
    y0 = 0.15*Ylm[0]+0.85*Ylm[1]
    ax.text(x0,y0, r"$s= $"+"%.2f" % m0+'$\pm$'+"%.2f" % dm0, fontsize=12)
    y0 = 0.30*Ylm[0]+0.70*Ylm[1]
    ax.text(x0,y0, r"$zp= $"+"%.2f" % b0+'$\pm$'+"%.2f" % db0, fontsize=12)    
    

    if ylabel: 
        if band!='w1' and band!='w2':
            ax.set_ylabel(r'$'+band+'^*$', fontsize=18) 
        elif band=='w1': 
            ax.set_ylabel(r'$W1^*$', fontsize=18)
        elif band=='w2':
            ax.set_ylabel(r'$W2^*$', fontsize=18)
            
    
    if xlabel:
        ax.set_xlabel(r'$Log( W_{mx}^i)$', fontsize=18) 
    
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

        
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    
    print len(pgc_), band, m0, b0
    
########################################################
def TFonline(clusterName='Coma Cluster', nest='Nest_100001', band='r'): 
    
    cluster = nest+'.csv'
    
    ctl   = np.genfromtxt(cluster , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
    PGC = ctl['PGC']   

    mask = np.isin(pgc, PGC)
    Wmxi_range = logWimx[mask]
    u_range = table['u'][mask]
    g_range = table['g'][mask]
    r_range = table['r'][mask]
    i_range = table['i'][mask]
    z_range = table['z'][mask]
    w1_range = table['w1'][mask]
    w2_range = table['w2'][mask]
    xLim_0 = min(Wmxi_range)-0.3
    xLim_1 = max(Wmxi_range)+0.2
    mags_range = np.concatenate((u_range,g_range,r_range,i_range,z_range,w1_range,w2_range))
    mags_range = mags_range[np.where(mags_range>0)]
    yLim_0 = max(mags_range)+0.7
    yLim_1 = min(mags_range)-0.7

    A = Redden[band]
    eA = eRed[band]
    rang=dye[band]
    mag = table[band] - A
    mag_e = np.sqrt((mag*0.+0.05)**2+eA**2)  
      
    pgc_ = []
    logWimx_ = []
    logWimx_e_ = []
    mag_ = []
    mag_e_ = []
    Sqlt_ = []
    Wqlt_ = []
    
    for i in range(len(pgc)):
        id=pgc[i]
        if id in PGC:
            pgc_.append(id)
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
        


    if band=='w1':
       band='W1'
    elif band=='w2':
       band='W2'

    hover = HoverTool(tooltips=[ 
        ("LogWimx", "@LogWimx"),
        (band, "@"+band),
        ("PGC", "@PGC"),
        ])

    hover.point_policy='snap_to_data'
    hover.line_policy='nearest'#'prev'

    TOOLS = ['pan', 'tap', 'wheel_zoom', 'box_zoom', 'reset', 'save']

    p = figure(tools=TOOLS, toolbar_location="below", plot_width=550, plot_height=450, title=clusterName+' - '+nest)
    p.title.text_font_size = '14pt'
    p.title.text_color = 'green'
    p.grid.grid_line_color="gainsboro"
        

    source = ColumnDataSource({'LogWimx': logWimx_, band: mag_, 'PGC': pgc_})
    render = p.circle('LogWimx', band, source=source, size=5, color=rang, alpha=0.7, hover_color='red', hover_alpha=1, hover_line_color='red',
                        
                        # set visual properties for selected glyphs
                        selection_fill_color=rang,

                        # set visual properties for non-selected glyphs
                        nonselection_fill_alpha=1,
                        nonselection_fill_color=rang,)




    indx, = np.where(Sqlt_>2)
    pgc_ = pgc_[indx]
    logWimx_= logWimx_[indx]
    logWimx_e_= logWimx_e_[indx]
    mag_= mag_[indx]
    mag_e_= mag_e_[indx]
    Sqlt_= Sqlt_[indx]
    Wqlt_= Wqlt_[indx]


    indx, = np.where(Wqlt_>2)
    pgc_ = pgc_[indx]
    logWimx_= logWimx_[indx]
    logWimx_e_= logWimx_e_[indx]
    mag_= mag_[indx]
    mag_e_= mag_e_[indx]
    Sqlt_= Sqlt_[indx]
    Wqlt_= Wqlt_[indx]            
                
                
    y_ = np.linspace(0,100,50)
    M,B,samples=linMC1D(mag_, logWimx_-2.5, logWimx_e_, nsteps=1000, ignore=200)
    m = M[0] ; me=0.5*(M[1]+M[2])
    b = B[0] ; be=0.5*(B[1]+B[2])
    m0 = 1./m
    b0 = -b/m
    dm0 = np.abs(me/m**2)
    db0 = np.sqrt((me*b)**2+(be*m)**2)/m**2
    x_, xu, xl = linSimul(samples, y_, size=500)
    p.line(m*y_+b+2.5, y_, line_width=1, color="black", line_dash='dashed')


    mytext = Label(x=20, y=290, text="s= "+"%.2f" % m0+'+-'+"%.2f" % dm0, text_color='black', text_font_size='12pt', x_units='screen', y_units='screen')
    p.add_layout(mytext)
    mytext = Label(x=20, y=260, text="zp= "+"%.2f" % b0+'+-'+"%.2f" % db0, text_color='black', text_font_size='12pt', x_units='screen', y_units='screen')
    p.add_layout(mytext)


    p.legend.location = "top_right"
    

    p.x_range = Range1d(xLim_0, xLim_1)
    p.y_range = Range1d(yLim_0, yLim_1)

    p.xaxis.axis_label = 'Log(W^i_mx)'
    p.yaxis.axis_label = band+'* [mag]'
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.grid.grid_line_color="gainsboro"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"


    code = """
        
        var index_selected = source.selected['1d']['indices'][0];
        var win = window.open("http://edd.ifa.hawaii.edu/cf4_photometry/get_sdss_cf4.php?pgc="+source.data['PGC'][index_selected]+"#t01", "EDDesn", "width=1200, height=1000");
        try {win.focus();} catch (e){}

    """

    taptool = p.select(type=TapTool)
    taptool.callback = CustomJS(args=dict(source=source), code=code)



    ttp = """
        <div>
            <div>
                <span style="font-size: 16px; color: blue;">PGC:</span>
                <span style="font-size: 16px; font-weight: bold;">@PGC{int}</span>
            </div>
            <div>
                <span style="font-size: 16px; color: blue;">LogWimx:</span>
                <span style="font-size: 16px; font-weight: bold;">@LogWimx</span>
            </div>  
            <div>
                <span style="font-size: 16px; color: blue;">"""+band+""":</span>
                <span style="font-size: 16px; font-weight: bold;">@"""+band+"""{0.2f}</span>
            </div>          
        </div>
    """


    hover = HoverTool(tooltips=ttp, renderers=[render])

    hover.point_policy='snap_to_data'
    hover.line_policy='nearest'
    #hover.mode='vline'
    p.add_tools(hover)   

    cross = CrosshairTool()
    #cross.dimensions='height'
    cross.line_alpha = 0.3
    cross.line_color = 'green'
    p.add_tools(cross) 
    
    script, div = components(p)
    script = '\n'.join(['' + line for line in script.split('\n')])
    
    return script, div

###################################################
def batchplot(clusterName='Coma Cluster', nest='Nest_100001'):

    figname = nest+'.png'
    htmlname =  nest+'.html'
    plot_array(clusterName=clusterName, nest=nest)
    script_u, div_u   = TFonline(clusterName=clusterName, nest=nest, band='u')
    script_g, div_g   = TFonline(clusterName=clusterName, nest=nest, band='g')
    script_r, div_r   = TFonline(clusterName=clusterName, nest=nest, band='r')
    script_i, div_i   = TFonline(clusterName=clusterName, nest=nest, band='i')
    script_z, div_z   = TFonline(clusterName=clusterName, nest=nest, band='z')
    script_w1, div_w1 = TFonline(clusterName=clusterName, nest=nest, band='w1')
    script_w2, div_w2 = TFonline(clusterName=clusterName, nest=nest, band='w2')

    head = """
    <!DOCTYPE html>
    <html lang="en"> <head> <meta charset="utf-8"> <title>Bokeh Plot</title> <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.13.0.min.css" type="text/css" /> <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.13.0.min.js"></script> <script type="text/javascript"> Bokeh.set_log_level("info"); </script> 
    
    <style>

body {
   
    min-width: 800px;
   
    }

._center {
    margin: auto;
    width: 75%;
    min-width: 600px;
    border: 0px solid #ccc;
    padding: 10px;
    margin-bottom:20px;
    margin-top:10px;
    margin-right:50px;
    margin-left:210px;
    text-align:justify;
}

#container {
   min-height:100%;
   position:relative;
}
#header {
   box-sizing: border-box;
   padding-left: 20px;
   background:#660000;
   height:25px;
   color: white;
   margin-bottom:5px;
   padding: 0px 5px 2px;
}
#body {
}
#footer {
   box-sizing: border-box;
   position:fixed;
   bottom:0;
   left: 0;
   width:100%;
   height:20px;
   background: #660000;
   font-size : 10pt;
   color:white;
}

a.footerlink{
   color:white;
   font-size : 10pt;
   text-decoration:underline;
}
a.footerlink:hover {
   color:yellow;
 
}


 .floating-menu {
    font-family: sans-serif;
    background:  #800000;
    padding: 5px;
    width: 200px;
    z-index: 100;
    position: fixed;
    margin-top: 20px;
  }
  .floating-menu a,
  .floating-menu h3 {
    font-size: 0.9em;
    display: block;
    margin: 0 0.5em;
    color: white;
  }
  .floating-menu a:hover{
      color:yellow;
      }
 
  .floating-menu li {
      color: white;
      }
  .floating-menu li:hover {
      color: yellow;
      }
 
 
  hr {
    display: block;
    height: 1px;
    border: 0;
    border-top: 1px solid #ccc;
    margin: 1em 0;
    padding: 0;
}
 
 
</style>


</header>
<body>

<div id="container">
<div id="body">

<nav class="navbar navbar-expand-lg navbar-dark bg-primary">
 


</nav>


  <nav class="floating-menu">
    <ul>
      <li><a title="NEST_100001" href="NEST_100001.html">Coma Cluster </a></li>
      <li><a title="NEST_100002" href="NEST_100002.html">Virgo Cluster </a></li>
      <li><a title="NEST_100003" href="NEST_100003.html">Centaurus Cluster </a></li>
      <li><a title="NEST_100005" href="NEST_100005.html">Abell 1367 </a></li>
      <li><a title="NEST_100006" href="NEST_100006.html">Hydra Cluster </a></li>
      <li><a title="NEST_100007" href="NEST_100007.html">Abell 2151 </a></li>
      <li><a title="NEST_100008" href="NEST_100008.html">Ursa Major </a></li>
      <li><a title="NEST_100014" href="NEST_100014.html">Antlia Cluster </a></li>
      <li><a title="NEST_100018" href="NEST_100018.html">NGC 4065 Cluster </a></li>
      <li><a title="NEST_100030" href="NEST_100030.html">Cancer Cluster </a></li>
      <li><a title="NEST_120002" href="NEST_120002.html">Virgo W Cluster </a></li>
      <li><a title="NEST_200003" href="NEST_200003.html">Abell 262 </a></li>
      <li><a title="NEST_200005" href="NEST_200005.html">NGC 410 Cluster </a></li>
      <li><a title="NEST_200006" href="NEST_200006.html">NGC 507 Cluster </a></li>
      <li><a title="NEST_200012" href="NEST_200012.html">Abell 400 </a></li>
      <li><a title="NEST_200015" href="NEST_200015.html">Fornax Cluster </a></li>
      <li><a title="NEST_200016" href="NEST_200016.html">Abell 2634/66 </a></li>
      <li><a title="NEST_200017" href="NEST_200017.html">Abell 539 </a></li>
      <li><a title="NEST_200037" href="NEST_200037.html">NGC 70 Cluster </a></li>
      <li><a title="NEST_200045" href="NEST_200045.html">NGC 80 Cluster </a></li>
      <li><a title="NEST_200092" href="NEST_200092.html">Pegasus Cluster </a></li>
    </ul>
  </nav>

<div class="_center">
    

    
    </head> <body>
    """

    tail = """
    
    </div>   <!-- center -->


</div>  <!-- body -->
<div id="footer">
<span> &#160;&#160;&#9400; Copyright 2018 Cosmicflows Team. Designed and supported by <a class="footerlink" href="http://www.kourkchi.com/contact.html" target="contactESN">Ehsan Kourkchi</a>.</span>
</div>   <!-- footer -->
</div>   <!-- container -->
    
    
    </body></html>"""

    with open(htmlname, "w") as text_file:
        text_file.write(head)
        
        text_file.write('<table><tr>')
        
        text_file.write('<td><img src="'+figname+'" width="500"></td>')
        
        text_file.write('<td>')
        if True:
            text_file.write('<table>')
            
            text_file.write('<tr><td>')
            text_file.write(div_u)
            text_file.write(script_u)   
            text_file.write('</td><td>')
            text_file.write(div_g)
            text_file.write(script_g)  
            text_file.write('</td></tr>')
            
            text_file.write('<tr><td>')
            text_file.write(div_r)
            text_file.write(script_r)   
            text_file.write('</td><td>')
            text_file.write(div_i)
            text_file.write(script_i)  
            text_file.write('</td></tr>')    
            
            text_file.write('<tr><td>')
            text_file.write(div_z)
            text_file.write(script_z)   
            text_file.write('</td><td>')
            text_file.write('</td></tr>')     

            text_file.write('<tr><td>')
            text_file.write(div_w1)
            text_file.write(script_w1)   
            text_file.write('</td><td>')
            text_file.write(div_w2)
            text_file.write(script_w2)  
            text_file.write('</td></tr>') 
                
            text_file.write('</table>')
        text_file.write('</td>') 
        
        text_file.write('</tr></table>')
        text_file.write(tail)

###################################################

batchplot(nest='NEST_100001', clusterName='Coma Cluster')
batchplot(nest='NEST_100002', clusterName='Virgo Cluster')
batchplot(nest='NEST_100005', clusterName='Abell 1367')
batchplot(nest='NEST_100007', clusterName='Abell 2151')
batchplot(nest='NEST_100008', clusterName='Ursa Major')
batchplot(nest='NEST_100018', clusterName='NGC 4065 Cluster')
batchplot(nest='NEST_100030', clusterName='Cancer Cluster')
batchplot(nest='NEST_120002', clusterName='Virgo W Cluster')
batchplot(nest='NEST_200003', clusterName='Abell 262')
batchplot(nest='NEST_200005', clusterName='NGC 410 Cluster')
batchplot(nest='NEST_200006', clusterName='NGC 507 Cluster')
batchplot(nest='NEST_200012', clusterName='Abell 400')
batchplot(nest='NEST_200016', clusterName='Abell 2634/66')
#batchplot(nest='NEST_200017', clusterName='Abell 539') # SOUTH
batchplot(nest='NEST_200037', clusterName='NGC 70 Cluster')
batchplot(nest='NEST_200045', clusterName='NGC 80 Cluster')
batchplot(nest='NEST_200092', clusterName='Pegasus Cluster')
#batchplot(nest='NEST_100003', clusterName='Centaurus Cluster') # SOUTH
#batchplot(nest='NEST_100006', clusterName='Hydra Cluster') # SOUTH
#batchplot(nest='NEST_100014', clusterName='Antlia Cluster') # SOUTH
#batchplot(nest='NEST_200015', clusterName='Fornax Cluster') # SOUTH




