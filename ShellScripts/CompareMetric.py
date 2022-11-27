from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import numpy as np
import math
import sys

TEST_DAY=sys.argv[1]
NumFlights=int(sys.argv[2])
NumColors=30
minutes=20

TrafficFactorGA=open("Website/backend/OutputFolder/TrafficFactor.txt")
MetricsGA=pd.read_csv("Website/backend/OutputFolder/AerialTimeGD.txt")
OpToFrontend=open("Website/backend/OutputFolder/OutputToFrontend.txt")

TrafficFactorReal = dict()
if(NumFlights==-1):
    TrafficFactorReal = pk.load(open(f"Dataset/{TEST_DAY}/{TEST_DAY}_KDict_FullDay.pkl", "rb"))
else:
    TrafficFactorReal = pk.load(open(f"ShellScripts/RealMetrics/{TEST_DAY}_KDict.pkl", "rb"))
RealMetrics=pd.read_csv(f"ShellScripts/RealMetrics/{TEST_DAY}-gd-aat.csv")
AirportTrafficDf=pd.read_csv(f"ShellScripts/RealMetrics/{TEST_DAY}_idif.csv")

convexDict=pk.load(open("Notebooks/Outputs/ConvexDict.pkl","rb"))
dayFlights=pd.read_csv(f"Dataset/{TEST_DAY}/{TEST_DAY}_flights.csv")
if(NumFlights!=-1):
    dayFlights=dayFlights[:NumFlights]



realAT=RealMetrics["Aerial Time"]
realGD=list(RealMetrics["ground_delay"])
realGD=list(map(lambda x:abs(x),realGD))

q25, q75 = np.percentile(realGD, [25, 75])
bin_width = 2 * (q75 - q25) * len(realGD) ** (-1/3)
binsRGD = round((max(realGD) - min(realGD)) / bin_width)

plt.figure(1)
plt.hist(realGD, range=(min(realGD),max(realGD)),color='b', bins=binsRGD)
plt.title('Real Ground Delay')
plt.xlabel('Real Ground Delay')
plt.ylabel('Frequency')
plt.savefig(f"OutputImages/RealGroundDelay_{TEST_DAY}.png")

GA_AT=MetricsGA["Aerial Time"]
GA_GD=MetricsGA["Ground Holding"]
absDiffAT=[]
GATrafFactor=defaultdict(int)
RealTrafFactor=defaultdict(int)
for line in TrafficFactorGA:
    GATrafFactor[int(float(line))]+=1
for key,value in TrafficFactorReal.items():
    RealTrafFactor[value]+=1
data={"SectorCount":[i for i in range(30)],
     "GA Model":[GATrafFactor[sector] for sector in range(30)],
     "Real World":[RealTrafFactor[sector] for sector in range(30)]}
SectorDataFrame=pd.DataFrame(data)
for i in range(len(GA_AT)):
    absDiffAT.append(realAT[i]-GA_AT[i])
q25, q75 = np.percentile(absDiffAT, [25, 75])
bin_width = 2 * (q75 - q25) * len(absDiffAT) ** (-1/3)
binsAT = round((max(absDiffAT) - min(absDiffAT)) / bin_width)

q25, q75 = np.percentile(GA_GD, [25, 75])
bin_width = 2 * (q75 - q25) * len(GA_GD) ** (-1/3)
if(bin_width==0):
    bin_width=1
binsGD = round((max(GA_GD) - min(GA_GD)) / bin_width)

plt.figure(2)
plt.hist(absDiffAT, range=(min(absDiffAT),max(absDiffAT)),color='b', bins=binsAT)
plt.title('Flight Time')
plt.xlabel('Difference in Flight Time between Real Data and GA Solution in minutes')
plt.ylabel('Frequency')

plt.savefig(f"OutputImages/AerialTime{TEST_DAY}.png")

plt.figure(3)
plt.hist(GA_GD, range=(min(GA_GD),max(GA_GD)),color='r', bins=binsGD)
plt.title('Delay in takeoff')
plt.xlabel('Ground Delay in minutes')
plt.ylabel('Frequency')

plt.savefig(f"OutputImages/GroundDelay{TEST_DAY}.png")


SectorDataFrame.plot.bar(x='SectorCount',figsize=(10,5),ylabel = 'Number of sectors')
plt.savefig(f"OutputImages/Traffic{TEST_DAY}.png")

RMSE_AT=0
for i in range(len(GA_AT)):
    RMSE_AT+=((realAT[i]-GA_AT[i])**2)
RMSE_AT/=len(GA_AT)
RMSE_AT=RMSE_AT**(0.5)
print("The RMSD error for Aerial Time is ",RMSE_AT)
print("Average Ground Delay is ",sum(GA_GD)/len(GA_GD))
print("Average Real Ground Delay is ",sum(realGD)/len(realGD))

def getColours(K):
    d=dict()
    r,g,b=255,255,255
    cut=255/K
    for i in range(K):
        d[i]=[r/255,g/255,b/255]
        r-=cut
        g-=cut
        b-=cut
    return d

fig=pk.load(open("Notebooks/Outputs/Simulator.pkl","rb"))
plt.figure(figsize=(20,20))
realData=TrafficFactorReal
hue=sorted([(k,v) for k,v in realData.items()],key=lambda x:x[1])
colors=getColours(NumColors)
avgReal=0
avgGA=0
seen=[False]*NumColors
for sector,k in hue:
    avgReal+=k
    if k >= NumColors:
        k=NumColors-1
    x,y=zip(*convexDict[sector])
    colour=colors[k]
    if(not seen[k]):
        plt.fill(x,y,facecolor=colour,label=k)
        seen[k]=True
    else:
        plt.fill(x,y,facecolor=colour,label=f"_{k}")
    plt.legend(prop={'size': 16})
avgReal/=1250
plt.savefig(f"OutputImages/RealTraffic{TEST_DAY}.jpg")
print(f"Average sector count in real data {avgReal}")

fig=pk.load(open("Notebooks/Outputs/Simulator.pkl","rb"))
plt.figure(figsize=(20,20))
TrafficFactorGA.seek(0)
gaData=TrafficFactorGA.readlines()
gaData=list(map(lambda x:x[:len(x) -1],gaData))
gaData=list(map(lambda x:int(x),gaData))
gaDict=dict()
for index,value in enumerate(gaData):
    gaDict[index]=value
hue=sorted([(k,v) for k,v in gaDict.items()],key=lambda x:x[1])
seen=[False]*NumColors
for sector,k in hue:
    avgGA+=k
    if k >= NumColors:
        k=NumColors-1
    x,y=zip(*convexDict[sector])
    colour=colors[k]
    if(not seen[k]):
        plt.fill(x,y,facecolor=colour,label=k)
        seen[k]=True
    else:
        plt.fill(x,y,facecolor=colour,label=f"_{k}")
    plt.legend(prop={'size': 16})
plt.savefig(f"OutputImages/GATraffic{TEST_DAY}.jpg")
avgGA/=1250
print(f"Average sector count in GA Solution {avgGA}")

def realTraffic(df,minutes):
    traffic=defaultdict(list)
    trafficMax=dict()
    sourceAirports=list(df["departure_airport_icao_code"])
    destinationAirports=list(df["arrival_airport_icao_code"])
    rangeSize=max(list(df["idif_actual_runway_arrival"]))
    numberOfRanges=math.ceil(rangeSize/minutes)
    departureTimes=list(df["idif_actual_runway_departure"])
    arrivalTimes=list(df["idif_actual_runway_arrival"])
    for i in range(len(departureTimes)):
        depart=int(departureTimes[i]/minutes)
        if len(traffic[sourceAirports[i]]) == 0:
            traffic[sourceAirports[i]]=[0 for _ in range(numberOfRanges + 1)]
        traffic[sourceAirports[i]][depart]+=1
        arrive=int(arrivalTimes[i]/minutes)
        if len(traffic[destinationAirports[i]]) == 0:
            traffic[destinationAirports[i]]=[0 for _ in range(numberOfRanges + 1)]
        traffic[destinationAirports[i]][arrive]+=1
    for airport,trafficRanges in traffic.items():
        trafficMax[airport]=max(trafficRanges)
    return trafficMax

def GATraffic(df,minutes,data):
    traffic=defaultdict(list)
    trafficMax=dict()
    sourceAirports=list(df["departure_airport_icao_code"])
    destinationAirports=list(df["arrival_airport_icao_code"])
    actualDeparture=[]
    actualArrival=[]
    for i in range(len(sourceAirports)):
        currentData=data[i].split(",")
        actualDeparture.append(int(currentData[-6]))
        actualArrival.append(int(currentData[-4]))
    rangeSize=max(actualArrival)
    numberOfRanges=math.ceil(rangeSize/minutes)
    for i in range(len(sourceAirports)):
        depart=int(actualDeparture[i]/minutes)
        if len(traffic[sourceAirports[i]]) == 0:
            traffic[sourceAirports[i]]=[0 for _ in range(numberOfRanges + 1)]
        traffic[sourceAirports[i]][depart]+=1
        arrive=int(actualArrival[i]/minutes)
        if len(traffic[destinationAirports[i]]) == 0:
            traffic[destinationAirports[i]]=[0 for _ in range(numberOfRanges + 1)]
        traffic[destinationAirports[i]][arrive]+=1
    for airport,trafficRanges in traffic.items():
        trafficMax[airport]=max(trafficRanges)
    return trafficMax

data=OpToFrontend.readlines()
x=realTraffic(AirportTrafficDf,minutes)
y=GATraffic(dayFlights,minutes,data)
gaSum=0
realSum=0
c=0
for airport in x:
    if y[airport] <= x[airport]:
        c+=1
    gaSum+=x[airport]
    realSum+=y[airport]
print(f"Traffic in GA is better than or equal to real data in {c*100/len(x)} % of the airports")
print("Real Average Airport Traffic ",realSum/c)
print("GA Average Airport Traffic ",gaSum/c)

