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


import json
#########################################################
## HTML header
head = """
<!DOCTYPE html>
<html lang="en"> 

<head> 
    <meta charset="utf-8"> <title>Bokeh Plot</title> <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.13.0.min.css" type="text/css
" /> <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.13.0.min.js"></script> <script type="text/javascript"> Bokeh.set_log_
level("info"); </script> 
</head>

    <body>

                
"""

#########################################################
## HTML footer
tail = """


    </body>
</html>
"""

#########################################################
def makeHTML(data, tileid='58570'):
    
    tileid=str(tileid)
    
    htmlName = 'tile-'+tileid+'.html'
    
    rootURL = 'http://www.astro.utah.edu/~u6022465/sandbox/'
    fitsURL = rootURL+'fits_files/tile-0'+tileid+'.fits'
    gaiaURL = rootURL+'hist_files/tile-'+tileid+'_maghist.png'
    tileURL = rootURL+'png_files/fiberassign_0'+tileid+'.png'
    
    with open(htmlName, "w") as text_file:
        
        
        text_file.write(head)
        
        
        text_file.write('<table><tr>')
        text_file.write('<td valign="top" width="500">')
        text_file.write('<table>')
        
        text_file.write('<tr><td><h1>Tile '+tileid+'</h1></td></tr>')
        text_file.write('<tr><td><div>')
        keylist=['assign_safe','assign_science','assign_sky','assign_std','assign_total','tile_dec','tile_obscond','tile_ra','unassigned']
        
        for key in keylist:#data[tileid]:
            text_file.write('<p><b>'+str(key)+':&nbsp;</b>'+str(data[tileid][key])+'</p>')
    
        text_file.write('</div></td></tr>')
        text_file.write('<tr><td><img src="'+gaiaURL+'" width="900"></td></tr>')
        text_file.write('<tr><td><h3>Download: <a href="'+fitsURL+'">tile-0'+tileid+'.fits</a></h3></td></tr>')
        text_file.write('</table>')
        text_file.write('</td>')
       
        text_file.write('<td><img src="'+tileURL+'" width="800"></td>') 
        text_file.write('</tr></table>') 
        
        text_file.write(tail)



#########################################################
with open("qa.json", "r") as read_file:
    data = json.load(read_file)
    idlist=list(data.keys())
    for tileid in idlist:

        makeHTML(data, tileid=tileid)
