from fastapi import FastAPI
from collections import deque,defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import distinctipy as ds
import pickle as pk
import pandas as pd
from io import BytesIO
from starlette.responses import StreamingResponse
app = FastAPI()
m = pk.load(open("../Notebooks/Outputs/M_ConversionMetric.pkl", "rb"))
ConvexDictRead = pk.load(open("../Notebooks/Outputs/ConvexDict.pkl", "rb"))
df=pd.read_csv("../Notebooks/Outputs/US_SectorMap.csv")
f = open('../Notebooks/Inputs/airports.json')
CentroidDict=pk.load(open("../Notebooks/Outputs/CentroidDict.pkl","rb"))
fig = plt.figure(pk.load(open("../Notebooks/Outputs/Simulator.pkl","rb")))
Paths=pk.load(open("../Notebooks/GA_Output.pkl","rb")) 

prevX=[deque() for _ in range(len(Paths))]
prevY=[deque() for _ in range(len(Paths))]
colors=ds.get_colors(len(Paths))
MaxSectorCountDict=defaultdict(int)



@app.get("/api")
def hello(position:int):
    tempDict=defaultdict(int)
    if position == 0:
        for pathIdx in range(len(Paths)):
            point=CentroidDict[Paths[pathIdx][position]]
            prevX[pathIdx].append(point[0])
            prevY[pathIdx].append(point[1])
            tempDict[Paths[pathIdx][position]]+=1
        plt.plot(prevX,prevY,'o',markersize=50,color=colors[pathIdx])
        fig.canvas.draw()
        plt.savefig(f'my_plot{position}.png')
    plt.show()
    return {"name":"Raghav"}
