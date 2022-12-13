from collections import deque,defaultdict
from scipy.spatial import distance as dst
from matplotlib import collections as mc
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import shapely.geometry as sp
import distinctipy as ds
import shapefile as shp
import pickle as pk
import pandas as pd
import numpy as np
import json
output = pk.load(open("../Notebooks/Outputs/ConvexDict.pkl", "rb"))
m=pk.load(open("../Notebooks/Outputs/M_ConversionMetric.pkl",'rb'))
centers=pd.read_csv("../Notebooks/Outputs/StateCentres.csv")
df=pd.read_csv("../Notebooks/Outputs/US_SectorMap.csv")
sf = shp.Reader('../Notebooks/Outputs/StateShapes/usa-states-census-2014.shp')
dpi=10
fig = plt.figure(figsize=(1000,1000),dpi=dpi)
fig.tight_layout()
for idd in range(0,len(sf.records())):
    shape_ex = sf.shape(idd)
    x_lon = np.zeros((len(shape_ex.points),1))
    y_lat = np.zeros((len(shape_ex.points),1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    xm_lon, ym_lat = m(x_lon, y_lat)
    plt.plot(xm_lon, ym_lat, c='black',linewidth=3)
    Name=sf.records()[idd][5]
    StateCenter=centers[centers["State"]==Name]
    lat=StateCenter["CenterLat"]
    lon=StateCenter["CenterLon"]
    x,y=m(lon,lat)
    plt.text(x,y,f"{sf.records()[idd][4]}",fontsize=75)
m.drawcoastlines(zorder=6)
sectorNumber=0
while sectorNumber<1250:
    x=output[sectorNumber]
    x1=list(map(lambda p:p[0],x))
    y1=list(map(lambda p:p[1],x))
    x2=list(df[df['sector']==sectorNumber]['easting'])
    y2=list(df[df['sector']==sectorNumber]['northing'])
    x3=sorted(x1)
    y3=sorted(y1)
    xmid=(x3[0]+x3[-1])/2
    ymid=(y3[0]+y3[-1])/2
    plt.plot(x1,y1,'r')
    plt.plot([x1[0],x1[-1]],[y1[0],y1[-1]],'r')
    sectorNumber+=1
fig.tight_layout()
fig.canvas.draw()
pk.dump(fig,open("../Notebooks/Outputs/Simulator.pkl","wb"))
pk.dump(fig,open("../Website/backend/SimFiles/Simulator.pkl","wb"))
