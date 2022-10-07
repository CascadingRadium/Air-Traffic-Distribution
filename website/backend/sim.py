from scipy.spatial import distance as dst
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import shapely.geometry as sp
import distinctipy as ds
import pickle as pk
import numpy as np
ConnectedSectorGraph = pk.load(open("ConnectedSectorGraph.pkl", "rb"))
Centroids = pk.load(open("CentroidDict.pkl", "rb"))
opfile=open("OutputToFrontend.txt","r")
fig = plt.figure(pk.load(open("Simulator.pkl","rb")))
ax = fig.axes[0]
def path_maker(pathFromGA,MpMSpeed,index):
    PointPath=[Centroids[pathFromGA[0]]] #To be changed in version 2.1
    Distance=0.0
    for SectorIdx in range(len(pathFromGA)-1):
        Cur=pathFromGA[SectorIdx]
        Next=pathFromGA[SectorIdx+1]
        PointPath.append(ConnectedSectorGraph.es[ConnectedSectorGraph.get_eid(Cur,Next)]["ConnectingPoint"])
        Distance+=dst.euclidean(PointPath[-2],PointPath[-1])
    PointPath.append(Centroids[pathFromGA[-1]])  #To be changed in version 2.1
    Distance+=dst.euclidean(PointPath[-2],PointPath[-1])
    res=[]
    color=index
    DistanceDelta = MpMSpeed
    PathLine = sp.LineString(PointPath)
    distances = np.arange(0, Distance, DistanceDelta)
    points = [PathLine.interpolate(distance) for distance in distances] + [PathLine.boundary.geoms[1]]
    return points
lines=opfile.readlines()
TimeDict=[[] for i in range(1440)]
toStart=1440
toStop=0
DistinctColors=ds.get_colors(len(lines))
for index,line in enumerate(lines):
    if(line[-1]=='\n'):
        line = line[:-1]
    SplitLine=line.split(',')
    ActualStartTime=int(SplitLine[-4])
    path=[int(i) for i in SplitLine[:len(SplitLine)-6]]
    MpMSpeed=float(SplitLine[-1])*30.8667
    points=path_maker(path,MpMSpeed,index)
    toStop=max(toStop,len(points)+ActualStartTime)
    toStart=min(toStart,ActualStartTime)
    TimeDict[ActualStartTime].append([(points[0].x,points[0].y),(points[0].x,points[0].y),index])
    for ptIdx in range(1,len(points)):
        TimeDict[ActualStartTime+ptIdx].append([(points[ptIdx-1].x,points[ptIdx-1].y),(points[ptIdx].x,points[ptIdx].y),index])
Started=False
toQuit=False
UserInfo="Left-Click to start the Simulator \nRight-Click to Quit"
plt.text(1200000,500000,UserInfo,fontsize = 120)
def StartSim(event):
    global CurTime
    global Started
    if(not Started and event.button == 1):
        Started=True
        title="Hour Minutes";
        plt.text(6000000,3600000,title,fontsize = 120)
        CurTime=toStart
        for CurTime in range(toStart,toStop+1):
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
            toPlotNow=[]
            c=[]
            for pointIndex in range(len(TimeDict[CurTime])):
                pointOne=TimeDict[CurTime][pointIndex][0]
                pointTwo=TimeDict[CurTime][pointIndex][1]
                color=TimeDict[CurTime][pointIndex][2]
                toPlotNow.append([pointOne,pointTwo])
                c.append(DistinctColors[color])
            if(len(toPlotNow)!=0):
                lc = mc.LineCollection(toPlotNow, colors=c, linewidths=30)
                ax.add_collection(lc)
            fig.canvas.draw()
            time.remove()
            plt.pause(0.05)
def SimStopper(event):
    global toQuit
    if(event.button == 3):
        if(not Started):
            plt.close()
        toQuit=True
id1 = fig.canvas.mpl_connect('button_press_event', StartSim)
id2 = fig.canvas.mpl_connect('button_press_event', SimStopper)
plt.show()
