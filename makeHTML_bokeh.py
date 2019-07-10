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

from make_SVobsplan_page import *


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
def makeHTML():
    
    
    htmlName = 'test.html'
    tileFile = 'ci-tiles-v7.fits'
    jsonFile="qa.json"

    
    with open(htmlName, "w") as text_file:
        
        text_file.write(head)
        
        text_file.write('<table width="100%"><tr>') # first row
        
        
        for month in [3,5,7]:  # generating 3 rows
            
            p = bokehTile(tileFile, jsonFile, TT=[0,0,0], DD=[2019,month,20])
            script, div = components(p)
            script = '\n'.join(['' + line for line in script.split('\n')])
            
            p2 = bokehTile(tileFile, jsonFile, TT=[0,0,0], DD=[2019,month+1,20])
            script2, div2 = components(p2)
            script2 = '\n'.join(['' + line for line in script2.split('\n')])        
            
            text_file.write('<td>') # column 1
            text_file.write(script); text_file.write(div)
            text_file.write('</td><td>') # column 2
            text_file.write(script2); text_file.write(div2)
            text_file.write('</td>')
            text_file.write('</tr>')
        
       
        text_file.write('</table>')
        
        text_file.write(tail)


#########################################################

makeHTML()
