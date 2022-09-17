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
for line in opfile:
	A=line.split(' ')
	x=list(A[0].split(','))
	x = [eval(i) for i in x]
	Paths.append(x)
	Times.append(int(A[1]))
print(Times)
position=0
prevX=[deque() for _ in range(len(Paths))]
prevY=[deque() for _ in range(len(Paths))]
colors=ds.get_colors(len(Paths))
MaxSectorCountDict=defaultdict(int)
def TraceFunction(event):
	global position
	global MaxSectorCountDict
	global axTab
	tempDict=defaultdict(int)
	data = {'Sector':[],
	'SectorCount':[],
	'MaxSectorCount':[]}
	if position == 0:
		for pathIdx in range(len(Paths)):
			point=CentroidDict[Paths[pathIdx][position]]
			prevX[pathIdx].append(point[0])
			prevY[pathIdx].append(point[1])
			tempDict[Paths[pathIdx][position]]+=1
		plt.plot(prevX,prevY,'o',markersize=50,color=colors[pathIdx])
		fig.canvas.draw()
		position+=1
	else:
		for pathIdx in range(len(Paths)):
			if(position<len(Paths[pathIdx])):
				tempDict[Paths[pathIdx][position]]+=1
				point=CentroidDict[Paths[pathIdx][position]]
				prevX[pathIdx].append(point[0])
				prevY[pathIdx].append(point[1])
				plt.plot(prevX[pathIdx],prevY[pathIdx],linewidth=25,color=colors[pathIdx])
				prevX[pathIdx].popleft()
				prevY[pathIdx].popleft()
		fig.canvas.draw()
		position+=1
		for key in tempDict :
			data['Sector'].append(key)
			data['SectorCount'].append(tempDict[key])
			MaxSectorCountDict[key]=max(MaxSectorCountDict[key],tempDict[key])
			data['MaxSectorCount'].append(MaxSectorCountDict[key])
cid = fig.canvas.mpl_connect('button_press_event', TraceFunction)
plt.show()
