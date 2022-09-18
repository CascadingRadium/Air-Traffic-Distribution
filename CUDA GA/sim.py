from collections import deque,defaultdict
import matplotlib.pyplot as plt
import distinctipy as ds
import pickle as pk
import pandas as pd
import json
CentroidDict=pk.load(open("CentroidDict.pkl","rb"))
fig = plt.figure(pk.load(open("Simulator.pkl","rb")))
opfile=open("OutputToSimulator.txt","r")
Paths=[]
Times=[]
MaxTime=0
for line in opfile:
	A=line.split(' ')
	x=list(A[0].split(','))
	x = [eval(i) for i in x]
	Paths.append(x)
	Times.append(int(A[1]))
	MaxTime=max(MaxTime,int(A[2]))
CurTime=0
prevX=[deque() for _ in range(len(Paths))]
prevY=[deque() for _ in range(len(Paths))]
colors=ds.get_colors(len(Paths))
FirstDict=defaultdict(list)
for path,startTime in enumerate(Times):
	FirstDict[startTime].append(path)
maxStartTime = max(Times)
FirstPlot=[True for _ in range(maxStartTime)]
position=0
def TraceFunction(event):
	global CurTime
	global axTab
	global position
	if(CurTime<=MaxTime):
		if(CurTime<maxStartTime and FirstPlot[CurTime]):
			for pathIdx in FirstDict[CurTime]:
				point=CentroidDict[Paths[pathIdx][0]]
				prevX[pathIdx].append(point[0])
				prevY[pathIdx].append(point[1])
			plt.plot([i for i in prevX if len(i)!=0],[i for i in prevY if len(i)!=0],'o',markersize=50,color=colors[pathIdx])
			fig.canvas.draw()
			FirstPlot[CurTime]=False
			CurTime+=1
		
cid = fig.canvas.mpl_connect('button_press_event', TraceFunction)
plt.show()
