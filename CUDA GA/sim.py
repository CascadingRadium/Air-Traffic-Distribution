from collections import deque,defaultdict
from matplotlib import collections  as mc
import matplotlib.pyplot as plt
import distinctipy as ds
import pickle as pk
import pandas as pd
import json
CentroidDict=pk.load(open("CentroidDict.pkl","rb"))
fig = plt.figure(pk.load(open("Simulator.pkl","rb")))
opfile=open("OutputToSimulator.txt","r")
NumODPairs=int(opfile.readline())
TimeDict=[]
for line in opfile:
	line=line[:len(line)-1]
	A=line.split(' ')
	row=[]
	for i in A:
		X=i.split(',')
		row.append((int(X[0]),int(X[1])))
	TimeDict.append(row)
CurTime=0
MaxTime=len(TimeDict)
colors=ds.get_colors(NumODPairs)
ax = fig.axes[0]
def TraceFunction(event):
	global CurTime
	global axTab
	toPlotNow=[]
	if(CurTime<=MaxTime):
		for 3pair in TimeDict[CurTime]:
			if(3pair[0]==3pair[1]):
				Point=CentroidDict[3pair[0]]
				ax.plot(Point[0],Point[1],'o',markersize=30,color=3pair[2])
			else:
				PointOne=CentroidDict[3pair[0]]
				PointTwo=CentroidDict[3pair[1]]
				toPlotNow.append([PointOne,PointTwo])
		if(len(toPlotNow)!=0):
			lc = mc.LineCollection(toPlotNow, colors=colors, linewidths=30)
			ax.add_collection(lc)
		fig.canvas.draw()
	CurTime+=1
cid = fig.canvas.mpl_connect('button_press_event', TraceFunction)
plt.show()
