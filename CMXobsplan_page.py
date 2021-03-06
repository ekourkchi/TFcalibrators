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
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Range1d, Label, TapTool, OpenURL, CustomJS, CrosshairTool, LinearAxis
import fitsio
from bokeh.io import output_notebook
import json
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import column
from bokeh.models import CustomJS
from bokeh.transform import linear_cmap
from bokeh.models import CustomJS, ColumnDataSource, DateSlider, DateRangeSlider
from datetime import datetime as dt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from datetime import datetime
from astropy.time import Time, TimezoneInfo
#output_notebook()


#### Functions
#########################################################
def add_plane(p,color='black', plane=None, projection=None):

  from kapteyn import wcs

  if plane==None or projection==None:
    return

  alpha = np.arange(0.,360,2)
  delta = alpha*0.

  tran = wcs.Transformation(plane + " ", projection)
  alpha, delta = tran((alpha,delta))

  #for i in range(len(alpha)):
  #  if alpha[i] >180:
  ind = np.argsort(alpha)
  alpha = alpha[ind]
  delta = delta[ind]

  p.line(alpha, delta, line_width=2, color=color)
  p.line(alpha+5, delta+5,  color='black', alpha=0.5)
  p.line(alpha-5, delta-5, color='black', alpha=0.5)
#########################################################
def skyCircle(tt, dd):
    
    ds = str(dd[0])+'-'+str(dd[1])+'-'+str(dd[2])
    ts = str(tt[0])+':'+str(tt[1])+':'+str(tt[2]) 
           
    observatory = "kpno"
    name = "Kitt Peak National Observatory"
    Lon =  -(111 + 35/60. + 59.6/3600)
    Lat = 31.9599
    Height = 2120.      # meter     


    obsTime = dt(dd[0],dd[1],dd[2],tt[0],tt[1],tt[2])
    reference = dt(2000,1,1,12,0,0)           # UTCref
    reference = time.mktime(reference.timetuple())
    obsTime   = time.mktime(obsTime.timetuple()) + 7*3600
    deltaT = (obsTime-reference) / (24*3600);

    # Convert to LST
    LST_hours = ((18.697374558 + 24.06570982441908 * deltaT) + Lon/15.) % 24;
    LST_degrees = LST_hours * 15


    obsTime = Time(ds+' '+ts)
    loc = EarthLocation(lat=Lat*u.deg, lon=Lon*u.deg, height=Height*u.m)

    zenithangle = 48.19  # deg The zenith angle at which airmass equals 1.5
    az = np.arange(0,360,3)
    alt = az*0 + (90 - zenithangle)

    newAltAzcoordiantes = SkyCoord(alt = alt, az = az, obstime = obsTime, frame = 'altaz', location = loc, unit="deg")
    ra = newAltAzcoordiantes.icrs.ra.deg
    dec = newAltAzcoordiantes.icrs.dec.deg

    newAltAzcoordiantes = SkyCoord(alt = [90], az = [90], obstime = obsTime, frame = 'altaz', location = loc, unit="deg")
    ra0 = newAltAzcoordiantes.icrs.ra.deg
    ra0 = ra0[0]
    ra0 = (ra-ra0)%360

    
    return ColumnDataSource({"RA0":ra0, "RA":(ra0+LST_degrees)%360, "DEC":dec})
#########################################################
def moonLoc(tt, dd):
    from astropy.coordinates import get_moon
    
    ds = str(dd[0])+'-'+str(dd[1])+'-'+str(dd[2])
    ts = str(tt[0])+':'+str(tt[1])+':'+str(tt[2])    
    t = Time(ds+' '+ts)


    
    loc = EarthLocation.of_site('Kitt Peak')
    moon_loc = get_moon(t, loc)      
    
    return moon_loc.ra, moon_loc.dec
#########################################################
def bokehTile(tileFile, jsonFile, TT=[0,0,0], DD=[2019,10,1],dynamic=False):


    citls,h=fitsio.read(tileFile,header=True)
    w=(np.where(citls['IN_DESI']==1)[0])
    inci=citls[w]

    with open(jsonFile, "r") as read_file:
        data = json.load(read_file)

    #########################################################
    ## Coloring scheme
    palette = ['green', 'red','orange']
    dye = []

    for tile in citls['TILEID']:

        rang =2 #'orange'
        if tile in inci['TILEID']:
              rang=2 #'orange'#2  # if  IN_DESI==1 for a tile in the tile file  
        #else:  rang = 3 # default color for all tiles in the CI tile file  
        if str(tile) in data:
            rang= 0#'green' #green default
            if len(data[str(tile)]['unassigned'])>0: # not assigned (red)
                rang = 1#'red' #'red' 
            if (0 in data[str(tile)]['gfa_stars_percam']):
                print(data[str(tile)]['gfa_stars_percam'])
                rang = 1#'cyan'
           
        dye.append(rang)

    #########################################################
    #TOOLS = ['pan', 'tap', 'wheel_zoom', 'box_zoom', 'reset', 'save']
    TOOLS = []
    
    obsTime = dt(DD[0],DD[1],DD[2],TT[0],TT[1],TT[2])
    
    p = figure(tools=TOOLS, toolbar_location="right", plot_width=800, plot_height=450,
              title="CMX-ALL SKY: Visible Sky at airmass<1.5 at KittPeak")#str(DD[1])+" - 2019")

    p.title.text_font_size = '16pt'
    p.title.text_color = 'black'
    p.grid.grid_line_color="gainsboro"

    ###############################  adding ecliptic plane+ hour grid ####################3
    add_plane(p, color='red', plane='ecliptic', projection='equatorial')


    mapper = linear_cmap(field_name='DYE', palette=palette, low=0 ,high=2)
    tiles = ColumnDataSource({'RA': citls['RA'], 'DEC': citls['DEC'], 'TILEID': citls['TILEID'],'EBV_MED':citls['EBV_MED'],'STAR_DENSITY':citls['STAR_DENSITY'],'DYE':dye})
    render = p.circle('RA', 'DEC', source=tiles, size=9, line_color='chocolate', color=mapper, alpha=0.4, hover_color='orange', hover_alpha=1, hover_line_color='red',

                        # set visual properties for selected glyphs
                        selection_fill_color='orange',
                        selection_line_color='orange',
                        # set visual properties for non-selected glyphs
                        nonselection_fill_alpha=0.4,
                        nonselection_fill_color=mapper)


    p.xaxis.axis_label = 'RA [deg]'
    p.yaxis.axis_label = 'Dec. [deg]'
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.grid.grid_line_color="gainsboro"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.x_range = Range1d(360,0)


    #mytext = Label(x=180, y=-35, text="S", text_color='gray', text_font_size='12pt') ; p.add_layout(mytext)
    #mytext = Label(x=180, y=88, text="N", text_color='gray', text_font_size='12pt') ; p.add_layout(mytext)
    #mytext = Label(x=350, y=45, text="E", text_color='gray', text_font_size='12pt', angle=np.pi/2) ; p.add_layout(mytext)
    #mytext = Label(x=4, y=45, text="W", text_color='gray', text_font_size='12pt', angle=np.pi/2) ; p.add_layout(mytext)


    ## Javascript code to open up custom html pages, once user click on a tile
    code = """
        var index_selected = source.selected['1d']['indices'][0];
        var tileID = source.data['TILEID'][index_selected];
        if (tileID!==undefined) {
        var win = window.open("http://www.astro.utah.edu/~u6022465/cmx/ALL_SKY/allSKY_ci_tiles/sub_pages/tile-"+tileID+".html", " ");
        try {win.focus();} catch (e){} }
    """

    taptool = p.select(type=TapTool)
    taptool.callback = CustomJS(args=dict(source=tiles), code=code)


    ## The html code for the hover window that contain tile infrormation
    ttp = """
        <div>
            <div>
                <span style="font-size: 16px; color: blue;">Tile ID:</span>
                <span style="font-size: 16px; font-weight: bold;">@TILEID{int}</span>
            </div>
            <div>
                <span style="font-size: 16px; color: blue;">RA:</span>
                <span style="font-size: 16px; font-weight: bold;">@RA</span>
            </div>  
            <div>
                <span style="font-size: 16px; color: blue;">Dec:</span>
                <span style="font-size: 16px; font-weight: bold;">@DEC</span>
            </div>     
            <div>
                <span style="font-size: 16px; color: blue;">EBV_MED:</span>
                <span style="font-size: 16px; font-weight: bold;">@EBV_MED</span>
            </div> 
            <div>
                <span style="font-size: 16px; color: blue;">STAR_DENSITY:</span>
                <span style="font-size: 16px; font-weight: bold;">@STAR_DENSITY</span>
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
    cross.line_color = 'gray'
    p.add_tools(cross)

    # Setting the second y axis range name and range
    p.extra_y_ranges = {"foo": p.y_range}
    p.extra_x_ranges = {"joo": p.x_range}

    #Adding the second axis to the plot.  
    p.add_layout(LinearAxis(y_range_name="foo"), 'right')
    p.add_layout(LinearAxis(x_range_name="joo"), 'above')

    if dynamic:        
      circleSource = skyCircle(TT, DD)
      p.circle('RA', 'DEC', source=circleSource, size=1.5, color='black')
    else:
        circleSource = skyCircle(TT, DD)
        p.circle('RA', 'DEC', source=circleSource, size=1.5, color=None)
        
    
    ### Dealing with the Moon
    inFile = 'moon_loc.csv'
    tbl_moon = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=np.float)
    m_ra, m_dec = moonLoc(TT, DD)
    moonSource = ColumnDataSource({"moon_RAS":tbl_moon['ra'], "moon_DECS":tbl_moon['dec']})
    moon_RADEC = ColumnDataSource({"moon_ra":[m_ra.deg], "moon_dec":[m_dec.deg]})
    
    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC, size=110, color='cyan',alpha=0.2)
    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC, size=4, color='blue')



    callback = CustomJS(args=dict(source=circleSource, source_moon=moonSource, source_moon_RADEC=moon_RADEC), code="""
                // First set times as if they were UTC
                var t = new Date(time_slider.value);
                var d = new Date(date_slider.value);
                var data = source.data;
                var ra = data['RA'];
                var ra0 = data['RA0'];
                
                var data_moon = source_moon.data;
                var ras_moon = data_moon['moon_RAS'];
                var decs_moon = data_moon['moon_DECS'];
                
                var moonRADEC = source_moon_RADEC.data;
                var moon_ra = moonRADEC['moon_ra'];
                var moon_dec = moonRADEC['moon_dec'];
                                
                var Hour  = t.getUTCHours();
                var Day   = d.getDate();
                var Month = d.getMonth();
                
                var Year = new Array(31,28,31,30,31,30,31,31,30,31,30,31);
                var all_FULdays = 0;
                for (var i = 0; i < Month; i++)
                    all_FULdays=all_FULdays+Year[i];
                all_FULdays = all_FULdays + (Day-1);
                
                if (Hour<12) all_FULdays=all_FULdays+1;
                
                var all_minutes = all_FULdays*24+Hour;
                
                if (all_minutes<8800) {
                    moon_ra[0] = ras_moon[all_minutes];
                    moon_dec[0] = decs_moon[all_minutes];   
                    //alert(Day+" "+Month+" "+all_minutes);            
                }
                                
                if (t.getUTCHours() < 12) {
                    d.setTime(date_slider.value + 24*3600*1000);
                } else {
                    d.setTime(date_slider.value);
                }
                d.setUTCHours(t.getUTCHours());
                d.setUTCMinutes(t.getUTCMinutes());
                d.setUTCSeconds(0);        
                
                // Correct to KPNO local time
                // d object still thinks in UTC, which is 7 hours ahead of KPNO
                d.setTime(d.getTime() + 7*3600*1000);
                // noon UT on 2000-01-01
                var reftime = new Date();
                reftime.setUTCFullYear(2000);
                reftime.setUTCMonth(0);   // Months are 0-11 (!)
                reftime.setUTCDate(1);    // Days are 1-31 (!)
                reftime.setUTCHours(12);
                reftime.setUTCMinutes(0);
                reftime.setUTCSeconds(0);
                
                // time difference in days (starting from milliseconds)
                var dt = (d.getTime() - reftime.getTime()) / (24*3600*1000);

                // Convert to LST
                var mayall_longitude_degrees = -(111 + 35/60. + 59.6/3600);
                var LST_hours = ((18.697374558 + 24.06570982441908 * dt) + mayall_longitude_degrees/15) % 24;
                var LST_degrees = LST_hours * 15;
                
                

                for (var i = 0; i < ra.length; i++) {
                    ra[i] = (ra0[i] + LST_degrees) % 360;
                    //alert(LST_degrees);
                }
                source.change.emit();
                source_moon_RADEC.change.emit();
                //alert(d);
    """)
    

    
    
    if dynamic:
        
        ### TIME
        Timeslider = DateSlider(start=dt(2019,1,1,16,0,0), end=dt(2019,1,2,8,0,0), value=dt(2019,1,2,0,0,0), step=1, title="KPNO local time(hh:mm)", format="%H:%M",width=800)
        
        ## DATE
        Dateslider = DateSlider(start=dt(2019,1,1,0,0,0), end=dt(2020,1,1,0,0,0), value=dt(2019,10,1,0,0,0), step=1, title="Date of sunset(4pm-8am)", format="%B:%d",width=800)

        callback.args['time_slider'] = Timeslider
        callback.args['date_slider'] = Dateslider

        Dateslider.js_on_change('value', callback)
        Timeslider.js_on_change('value', callback)

        layout = column(p, Dateslider, Timeslider)
        #show(p)
        return layout
    
    return p

#########################################################

if __name__ == "__main__":
    
    json_path = '/uufs/chpc.utah.edu/common/home/bolton-group1/bolton_data2/kdawson/sarahE/cmx/ALL_SKY/allSKY_CItiles/FAout_citiles_withgfa_dr8+supp_newGFAcat/'
    tiles_path='/uufs/chpc.utah.edu/common/home/bolton-group1/bolton_data2/kdawson/sarahE/cmx/ALL_SKY/allSKY_CItiles/'
    p = bokehTile(tileFile=tiles_path+'ci_tiles_ALLSKY.fits', jsonFile=json_path+"qa.json", TT=[0,0,0], DD=[2019,10,1],dynamic=True)
    
    script, div = components(p)
    script = '\n'.join(['' + line for line in script.split('\n')])
    show(p)

    
