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
from scipy.optimize import curve_fit
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
########################################################
def linfit(x, m, b):
    return m * x + b
def linfit_b(x, m):
    return m * x
def linfit_m(x, b):
    return x+b
########################################################
table   = np.genfromtxt('corrected_mags_linewidth_all.csv' , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
pgc = table['pgc']

########################################################
def plot_array(table, reject=[], weird=[], clusterName='Coma Cluster', nest='Nest_100001', 
               xLim = (1.5,2.9), yLim = (19.5,6.5), isVirgo=False):
    
    cluster = nest+'.csv'
    figname = nest+'.png'
    
    ctl   = np.genfromtxt(cluster , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
    PGC = ctl['PGC']
    
    dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple", 'w2': "brown"}
    fig = py.figure(figsize=(5, 18), dpi=100) 
    fig.subplots_adjust(wspace=0, top=0.98, bottom=0.04, left=0.17, right=0.98)
    gs = gridspec.GridSpec(7, 1) ; p=0
    
    band_lst = ['u', 'g','r','i','z','w1','w2']
    
    for band in band_lst:
        
        xlabel = False; ylabel=True
        if band=='w2': xlabel=True
        
        ax = plt.subplot(gs[p]) ; p+=1
        
        try: 
            plot_band(ax, table, PGC, reject=reject, weird=weird, color=dye[band], 
                  xlabel=xlabel, ylabel=ylabel, band=band, xLim=xLim, yLim=yLim, isVirgo=isVirgo)
        except:
            p-=1
            pass
        
        yticks = ax.yaxis.get_major_ticks()
#         if band!='w2': yticks[-1].label1.set_visible(False)
        if band!='w2': plt.setp(ax.get_xticklabels(), visible=False)   
        if band=='u': ax.set_title(nest+' ... '+clusterName, fontsize=14)
        
    
    plt.subplots_adjust(hspace=.0, wspace=0)
    fig.savefig(figname)
########################################################
def plot_band(ax, table, PGC, reject=[], weird=[], color='red', 
              xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band='r', 
              xLim = (1.5,2.9), yLim = (19.5,6.5), isVirgo=False):  
    
    pgc       = table['pgc']
    logWimx   = table['logWimx']
    logWimx_e = table['logWimx_e']
    Vhel      = table['Vhel']
    
    Aj_e2     = getReddening_err(table, band1=band, band2='w2')
    
    mag = table[band+'_sss']
    mag_e = np.sqrt((mag*0.+0.05)**2+Aj_e2**2)
    
    pgc_ = []
    logWimx_ = []
    logWimx_e_ = []
    mag_ = []
    mag_e_ = []
    Sqlt_ = []
    Wqlt_ = []
    
    for i, idd in enumerate(pgc):
        if idd in PGC and not idd in reject:
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
        
        if pgc_[i] in weird:  ### open circle symbols
                ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], yerr=mag_e_[i], fmt='o', 
                            color='k', markersize=3, markerfacecolor='white')   
        else:
            if not band in ['w1','w2'] and Sqlt_[i]>=2:
                ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], yerr=mag_e_[i], fmt='o', 
                            color=color, markersize=4, alpha=0.7)
                dofit[i]=1
            elif band in ['w1','w2'] and Wqlt_[i]>=2:
                ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], yerr=mag_e_[i], fmt='o', 
                            color=color, markersize=4, alpha=0.7)
                dofit[i]=1
            else:
                ax.errorbar(logWimx_[i], mag_[i], xerr=logWimx_e_[i], yerr=mag_e_[i], fmt='o', 
                            color='k', markersize=3, markerfacecolor='white')   

    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.5, direction='in')
    ax.minorticks_on()
    ax.set_xlim(xLim)        
    ax.set_ylim(yLim)
    
    
    indx, = np.where(dofit==1)
    pgc_ = pgc_[indx]
    logWimx_= logWimx_[indx]
    logWimx_e_= logWimx_e_[indx]
    mag_= mag_[indx]
    mag_e_= mag_e_[indx]
    
    ### 'Virgo W cluster' condition
    if band in ['u', 'g']:
        indx = np.asarray([int(i) for i, idd in enumerate(pgc_) if not idd in [39886, 39114]])
        logWimx_= logWimx_[indx]
        logWimx_e_= logWimx_e_[indx]
        mag_= mag_[indx]
        mag_e_= mag_e_[indx]            
    
    
   
    fit, cov = curve_fit(linfit, mag_, logWimx_-2.5, sigma=np.sqrt(logWimx_e_**2+0.01**2))
    m, b = fit
    me = np.sqrt(cov[0,0])
    be = np.sqrt(cov[1,1])
    mbe = cov[1,0]


    y = np.linspace(5,20,50)
    x = m*y+b+2.5
    ax.plot(x,y, 'k--')  
    m0 = 1./m
    b0 = -b/m
    dm0 = np.abs(me/m**2)
    db0 = np.sqrt((b0**2 )*(be**2/b**2+me**2/m**2-2*mbe/m/b))
    
    Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
    x0 = 0.95*Xlm[0]+0.05*Xlm[1]
    y0 = 0.15*Ylm[0]+0.85*Ylm[1]
    ax.text(x0,y0, "Slope = "+"%.2f" % m0+'$\pm$'+"%.2f" % dm0, fontsize=12, color='k')
    y0 = 0.30*Ylm[0]+0.70*Ylm[1]
    ax.text(x0,y0, "ZP = "+"%.2f" % b0+'$\pm$'+"%.2f" % db0, fontsize=12, color='k')     
    

#### Linear MCMC fit, considering all errors in both directions
    y_ = np.linspace(5,20,50)
    M,B,samples=linMC1D(mag_, logWimx_-2.5, np.sqrt(logWimx_e_**2+0.01**2), nsteps=1000, ignore=200)
    m = M[0] ; me=0.5*(M[1]+M[2])
    b = B[0] ; be=0.5*(B[1]+B[2])
    m0 = 1./m
    b0 = -b/m
    dm0 = np.abs(me/m**2)
    db0 = np.sqrt((me*b)**2+(be*m)**2)/m**2
    x_, xu, xl = linSimul(samples, y_, size=500)
    ax.fill_betweenx(y_, x_+3*xu+2.50, x_-3*xl+2.50, color='k', alpha=0.2, edgecolor="none")

   
    if ylabel: 
        if not band in ['w1','w2']:
            ax.set_ylabel(r'$'+band+'^*$', fontsize=18) 
        else:
            ax.set_ylabel(r'$'+band.upper()+'^*$', fontsize=18)
    
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
########################################################
def TFonline(table, clusterName='Coma Cluster', nest='Nest_100001', band='r', reject=[], weird=[], isVirgo=False): 
    
    cluster = nest+'.csv'
    ctl   = np.genfromtxt(cluster , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
    PGC = ctl['PGC']
    
    pgc       = table['pgc']
    logWimx   = table['logWimx']
    logWimx_e = table['logWimx_e']
    Vhel      = table['Vhel']
    
    Aj_e2     = getReddening_err(table, band1=band, band2='w2')
    
    mag = table[band+'_sss']
    mag_e = np.sqrt((mag*0.+0.05)**2+Aj_e2**2)
    
    pgc_ = []
    logWimx_ = []
    logWimx_e_ = []
    mag_ = []
    mag_e_ = []
    Sqlt_ = []
    Wqlt_ = []
    
    for i, idd in enumerate(pgc):
        if idd in PGC and not idd in reject:
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
    
    
    dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple", 'w2': "brown"}
    rang=dye[band]         
                        
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

    dofit = np.zeros(N)
    
    for i in range(N):
        
        if not pgc_[i] in weird: 
            if not band in ['W1','W2'] and Sqlt_[i]>=2:
                dofit[i]=1
            elif band in ['W1','W2'] and Wqlt_[i]>=2:
                dofit[i]=1

    indx, = np.where(dofit==1)
    pgc_ = pgc_[indx]
    logWimx_= logWimx_[indx]
    logWimx_e_= logWimx_e_[indx]
    mag_= mag_[indx]
    mag_e_= mag_e_[indx]
    
    ### 'Virgo W cluster' condition
    if band in ['u', 'g']:
        indx = np.asarray([int(i) for i, idd in enumerate(pgc_) if not idd in [39886, 39114]])
        logWimx_= logWimx_[indx]
        logWimx_e_= logWimx_e_[indx]
        mag_= mag_[indx]
        mag_e_= mag_e_[indx]             
                
                
    y_ = np.linspace(0,100,50)
    fit, cov = curve_fit(linfit, mag_, logWimx_-2.5, sigma=np.sqrt(logWimx_e_**2+0.01**2))
    m, b = fit
    me = np.sqrt(cov[0,0])
    be = np.sqrt(cov[1,1])
    mbe = cov[1,0]
    m0 = 1./m
    b0 = -b/m
    dm0 = np.abs(me/m**2)
    db0 = np.sqrt((b0**2 )*(be**2/b**2+me**2/m**2-2*mbe/m/b))
    
    p.line(m*y_+b+2.5, y_, line_width=1, color="black", line_dash='dashed')


    mytext = Label(x=20, y=290, text="Slope= "+"%.2f" % m0+'+-'+"%.2f" % dm0, text_color='black', text_font_size='12pt', x_units='screen', y_units='screen')
    p.add_layout(mytext)
    mytext = Label(x=20, y=260, text="ZP= "+"%.2f" % b0+'+-'+"%.2f" % db0, text_color='black', text_font_size='12pt', x_units='screen', y_units='screen')
    p.add_layout(mytext)


    p.legend.location = "top_right"
    

    p.x_range = Range1d(1.5,2.9)
    p.y_range = Range1d(19.5,6.5)

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
    
    print nest, band
    
    return script, div

###################################################
def batchplot(table, clusterName='Coma Cluster', nest='Nest_100001', reject=[], weird=[], isVirgo=False):

    figname = nest+'.png'
    htmlname =  nest+'.html'
    
    plot_array(table, clusterName=clusterName, nest=nest, reject=reject, weird=weird, isVirgo=isVirgo)
    
    row = [False, False, False, False]
    
    try:
        script_u, div_u   = TFonline(table, clusterName=clusterName, nest=nest, band='u', reject=reject, weird=weird, isVirgo=isVirgo)
        row[0] = True
    except:
        script_u='' ; div_u=''
    try:
        script_g, div_g   = TFonline(table, clusterName=clusterName, nest=nest, band='g', reject=reject, weird=weird, isVirgo=isVirgo)
        row[0] = True
    except:
        script_g='' ; div_g=''        
    try:
        script_r, div_r   = TFonline(table, clusterName=clusterName, nest=nest, band='r', reject=reject, weird=weird, isVirgo=isVirgo)
        row[1] = True
    except:
        script_r='' ; div_r=''        
    try:
        script_i, div_i   = TFonline(table, clusterName=clusterName, nest=nest, band='i', reject=reject, weird=weird, isVirgo=isVirgo)
        row[1] = True
    except:
        script_i='' ; div_i=''        
    try:
        script_z, div_z   = TFonline(table, clusterName=clusterName, nest=nest, band='z', reject=reject, weird=weird, isVirgo=isVirgo)
        row[2] = True
    except:
        script_z='' ; div_z=''        
    try:
        script_w1, div_w1 = TFonline(table, clusterName=clusterName, nest=nest, band='w1', reject=reject, weird=weird, isVirgo=isVirgo)
        row[3] = True
    except:
        script_w1='' ; div_w1=''         
    try:
        script_w2, div_w2 = TFonline(table, clusterName=clusterName, nest=nest, band='w2', reject=reject, weird=weird, isVirgo=isVirgo)
        row[3] = True
    except:
        script_w2='' ; div_w2=''        

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
        
        text_file.write('<td valign="top">')
        if True:
            text_file.write('<table>')
            
            if row[0]:
                text_file.write('<tr><td>')
                text_file.write(div_u)
                text_file.write(script_u)   
                text_file.write('</td><td>')
                text_file.write(div_g)
                text_file.write(script_g)  
                text_file.write('</td></tr>')
            if row[1]:            
                text_file.write('<tr><td>')
                text_file.write(div_r)
                text_file.write(script_r)   
                text_file.write('</td><td>')
                text_file.write(div_i)
                text_file.write(script_i)  
                text_file.write('</td></tr>')    
            if row[2]:            
                text_file.write('<tr><td>')
                text_file.write(div_z)
                text_file.write(script_z)   
                text_file.write('</td><td>')
                text_file.write('</td></tr>')     
            if row[3]: 
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



reject = [43164,44405,93666]
weird = [43511]
batchplot(table, nest='NEST_100001', clusterName='Coma Cluster', reject=reject, weird=weird)


reject = []
weird = [41440]
batchplot(table, reject=reject, weird=weird, 
           clusterName='Virgo Cluster', nest='NEST_100002', isVirgo=True)


### SOUTH
reject = []
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Centaurus Cluster', nest='NEST_100003')


reject = [36323,36328,36330,36608,200155]
weird = [37140]
batchplot(table, reject=reject, weird=weird, 
           clusterName='Abell 1367', nest='NEST_100005')



### SOUTH
reject = []
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Hydra Cluster', nest='NEST_100006')


reject = [56977,2790835]
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Abell 2151 (Hercules)', nest='NEST_100007')



reject = [37550]
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Ursa Major', nest='NEST_100008')


### SOUTH
reject = []
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Antila Cluster', nest='NEST_100014')


reject = [38333]
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='NGC4065 Cluster', nest='NEST_100018')



reject = [23308]
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Cancer Cluster', nest='NEST_100030')


reject = [] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Virgo W Cluster', nest='NEST_120002')


reject = [] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Abell 262', nest='NEST_200003')


reject = [3446,4020] 
weird = [1904373]
batchplot(table, reject=reject, weird=weird, 
           clusterName='NGC410 Cluster', nest='NEST_200005')



reject = [4740,4876,5008] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='NGC507 Cluster', nest='NEST_200006')

### SOUTH
reject = [] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Fornax Cluster', nest='NEST_200015')


reject = [11150,11199,138562,3647754] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Abell 400', nest='NEST_200012')

reject = [85526,85643,90431,197699] 
weird = [5057398]
batchplot(table, reject=reject, weird=weird, 
           clusterName='Abell 2634/66', nest='NEST_200016')

### SOUTH
reject = [] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Abell 539', nest='NEST_200017')

reject = [1724] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='NGC70 Cluster', nest='NEST_200037')


reject = [90474] 
weird = [1707532]
batchplot(table, reject=reject, weird=weird, 
           clusterName='NGC80 Cluster', nest='NEST_200045')


reject = [70712, 70998, 71360] 
weird = []
batchplot(table, reject=reject, weird=weird, 
           clusterName='Pegasus Cluster', nest='NEST_200092')





