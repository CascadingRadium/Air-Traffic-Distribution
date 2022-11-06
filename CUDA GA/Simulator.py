from scipy.spatial import distance as dst
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import shapely.geometry as sp
import distinctipy as ds
import pickle as pk
import numpy as np
import pandas as pd
import sys
ConnectedSectorGraph = pk.load(open("SimFiles/ConnectedSectorGraph.pkl", "rb"))
opfile=open("OutputFolder/OutputToFrontend.txt","r")
fig = plt.figure(pk.load(open("SimFiles/Simulator.pkl","rb")))
airportCoords = pk.load(open("SimFiles/airportCoordDict.pkl","rb"))
ax = fig.axes[0]
SCALE_FACTOR=int(sys.argv[1])
def path_maker(pathFromGA,MpMSpeed,index,Src,Dst):
    PointPath=[airportCoords[Src]]
    Distance=0.0
    for SectorIdx in range(len(pathFromGA)-1):
        Cur=pathFromGA[SectorIdx]
        Next=pathFromGA[SectorIdx+1]
        PointPath.append(ConnectedSectorGraph.es[ConnectedSectorGraph.get_eid(Cur,Next)]["ConnectingPoint"])
        Distance+=dst.euclidean(PointPath[-2],PointPath[-1])
    PointPath.append(airportCoords[Dst])
    Distance+=dst.euclidean(PointPath[-2],PointPath[-1])
    res=[]
    color=index
    DistanceDelta = MpMSpeed
    PathLine = sp.LineString(PointPath)
    distances = np.arange(0, Distance, DistanceDelta)
    points = [PathLine.interpolate(distance) for distance in distances] + [PathLine.boundary.geoms[1]]
    return points
lines=opfile.readlines()
TimeDict=[[] for i in range(2880)]
toStart=2880
toStop=0
DistinctColors=ds.get_colors(len(lines))
airports=set()
for index,line in enumerate(lines):
    if(line[-1]=='\n'):
        line = line[:-1]
    SplitLine=line.split(',')
    airports.add(SplitLine[-1])
    airports.add(SplitLine[-2])
    ActualStartTime=int(SplitLine[-6])
    path=[int(i) for i in SplitLine[:len(SplitLine)-8]]
    MpMSpeed=float(SplitLine[-3])*30.8667
    points=path_maker(path,MpMSpeed,index,SplitLine[-2],SplitLine[-1])
    toStop=max(toStop,len(points)+ActualStartTime)
    toStart=min(toStart,ActualStartTime)
    TimeDict[ActualStartTime].append([(points[0].x,points[0].y),(points[0].x,points[0].y),index])
    for ptIdx in range(1,len(points)):
        TimeDict[ActualStartTime+ptIdx].append([(points[ptIdx-1].x,points[ptIdx-1].y),(points[ptIdx].x,points[ptIdx].y),index])
ICAO=list(airports)
xAir=[]
yAir=[]
for air in ICAO:
    xAir.append(airportCoords[air][0])
    yAir.append(airportCoords[air][1])
plt.scatter(xAir,yAir,marker='o' , s=3000)
for i, txt in enumerate(ICAO):
    plt.annotate(txt, (xAir[i], yAir[i]),fontsize=70,color ="red")
Started=False
toQuit=False
UserInfo="Left-Click to start the Simulator \nRight-Click to Quit"
plt.text(1200000,500000,UserInfo,fontsize = 120)
toPlotNow=[]
CurTime=toStart
for i in range(toStop-toStart+1):
    c=[]
    mclc=[]
    for pointIndex in range(len(TimeDict[CurTime])):
        pointOne=TimeDict[CurTime][pointIndex][0]
        pointTwo=TimeDict[CurTime][pointIndex][1]
        color=TimeDict[CurTime][pointIndex][2]
        mclc.append([pointOne,pointTwo])
        c.append(DistinctColors[color])
    lc = mc.LineCollection(mclc, colors=c, linewidths=30)
    toPlotNow.append(lc)
    CurTime+=1
def StartSim(event):
    global CurTime
    global Started
    global toPlotNow
    if(not Started and event.button == 1):
        Started=True
        title="Hour Minutes";
        plt.text(6000000,3650000,title,fontsize = 120)
        CurTime=toStart
        plotIndex=0
        for CurTime in range(toStart,toStop+1,SCALE_FACTOR):
            if(toQuit):
                plt.close()
                break
            time=""
            if(CurTime//60<10):
                time+=f"0{CurTime//60}:"
            else:
                time+=f"{CurTime//60}:"
            if(CurTime%60<10):
                time+=f"0{CurTime%60}"
            else:
                time+=f"{CurTime%60}"
            time=plt.text(6000000, 3500000, time, fontsize = 220)
            for idx in range(plotIndex,min(plotIndex+SCALE_FACTOR,len(toPlotNow))):
                ax.add_collection(toPlotNow[idx])
            plotIndex+=SCALE_FACTOR
            fig.canvas.draw()
            time.remove()
            plt.pause(0.005)
def SimStopper(event):
    global toQuit
    if(event.button == 3):
        if(not Started):
            plt.close()
        toQuit=True
id1 = fig.canvas.mpl_connect('button_press_event', StartSim)
id2 = fig.canvas.mpl_connect('button_press_event', SimStopper)
plt.show()
