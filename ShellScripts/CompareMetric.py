from collections import Counter
from collections import defaultdict
import matplotlib.ticker as mtick
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

Airspacefile=open("Website/backend/OutputFolder/AirspaceTraffic.txt")
Airportfile=open("Website/backend/OutputFolder/AirportTraffic.txt")
OpToFrontend=open("Website/backend/OutputFolder/OutputToFrontend.txt")
MetricsGA=pd.read_csv("Website/backend/OutputFolder/AerialTimeGD.txt")

convexDict=pk.load(open("Notebooks/Outputs/ConvexDict.pkl","rb"))

AirspaceTrafficReal = []
RealMetrics={}
AirportTrafficReal = dict()
if(NumFlights==-1):
    AirspaceTrafficReal = pk.load(open(f"Dataset/{TEST_DAY}/{TEST_DAY}_SectorTimeDict_FullDay.pkl", "rb"))
    RealMetrics=pd.read_csv(f"Dataset/{TEST_DAY}/{TEST_DAY}-gd-aat_FullDay.csv")
    AirportTrafficReal = pk.load(open(f"Dataset/{TEST_DAY}/{TEST_DAY}_AirportTrafficDict_FullDay.pkl", "rb"))
else:
    AirspaceTrafficReal = pk.load(open(f"ShellScripts/RealMetrics/{TEST_DAY}_SectorTimeDict.pkl", "rb"))
    RealMetrics=pd.read_csv(f"ShellScripts/RealMetrics/{TEST_DAY}-gd-aat.csv")
    AirportTrafficReal = pk.load(open(f"ShellScripts/RealMetrics/{TEST_DAY}_AirportTrafficDict.pkl", "rb"))

dayFlights=pd.read_csv(f"Dataset/{TEST_DAY}/{TEST_DAY}_flights.csv")
if(NumFlights!=-1):
    dayFlights=dayFlights[:NumFlights]

plt.figure(1)
realGD=list(RealMetrics["ground_delay"])
realGD=list(map(lambda x:abs(x),realGD))
q25, q75 = np.percentile(realGD, [25, 75])
bin_width = 2 * (q75 - q25) * len(realGD) ** (-1/3)
binsRGD = round((max(realGD) - min(realGD)) / bin_width)
plt.hist(realGD, range=(min(realGD),max(realGD)),color='b', bins=binsRGD)
plt.title('Real Ground Delay')
plt.xlabel('Real Ground Delay')
plt.ylabel('Frequency')
plt.savefig(f"OutputImages/RealGroundDelay_{TEST_DAY}.png")
print("Average Real Ground Delay is ",sum(realGD)/len(realGD))

plt.figure(2)
GA_GD=MetricsGA["Ground Holding"]
q25, q75 = np.percentile(GA_GD, [25, 75])
bin_width = 2 * (q75 - q25) * len(GA_GD) ** (-1/3)
if(bin_width==0):
    bin_width=1
binsGD = round((max(GA_GD) - min(GA_GD)) / bin_width)
plt.hist(GA_GD, range=(min(GA_GD),max(GA_GD)),color='r', bins=binsGD)
plt.title('Delay in takeoff')
plt.xlabel('Ground Delay in minutes')
plt.ylabel('Frequency')
plt.savefig(f"OutputImages/GroundDelay{TEST_DAY}.png")
print("Average GA Ground Delay is ",sum(GA_GD)/len(GA_GD))

plt.figure(3)
GA_AT=MetricsGA["Aerial Time"]
realAT=list(RealMetrics["Aerial Time"])
absDiffAT=[]
for i in range(len(GA_AT)):
    absDiffAT.append(realAT[i]-GA_AT[i])
q25, q75 = np.percentile(absDiffAT, [25, 75])
bin_width = 2 * (q75 - q25) * len(absDiffAT) ** (-1/3)
binsAT = round((max(absDiffAT) - min(absDiffAT)) / bin_width)
plt.hist(absDiffAT, range=(min(absDiffAT),max(absDiffAT)),color='b', bins=binsAT)
plt.title('Flight Time')
plt.xlabel('Difference in Flight Time between Real Data and GA Solution in minutes')
plt.ylabel('Frequency')
plt.savefig(f"OutputImages/AerialTime{TEST_DAY}.png")
RMSE_AT=0
for i in range(len(GA_AT)):
    RMSE_AT+=abs((realAT[i]-GA_AT[i]))
RMSE_AT/=len(GA_AT)
print("The average difference between real data and GA solution for Aerial Time is ",RMSE_AT)



plt.figure(4)
GATrafFactor=dict()
RealTrafFactor=dict()
for sector in range(1250):
    RealTrafFactor[sector]=max(AirspaceTrafficReal[sector])
AirspaceMatrix=Airspacefile.read().splitlines()
sectorNumber=0
SectorOccupancy=dict()
for line in AirspaceMatrix:
    linelist=line.split(',')
    linelist = [ int(x) for x in linelist ]
    GATrafFactor[sectorNumber]=max(linelist)
    noZeroList=[]
    for i in linelist:
        if(i>0):
            noZeroList.append(i)
        if(len(noZeroList)==0):
            SectorOccupancy[sectorNumber]=0
        else:
            SectorOccupancy[sectorNumber]=sum(noZeroList)/len(noZeroList)
    sectorNumber+=1
GA_Counter=defaultdict(int)
Real_Counter=defaultdict(int)
for key,value in GATrafFactor.items():
    GA_Counter[value]+=1
for key,value in RealTrafFactor.items():
    Real_Counter[value]+=1
data={"SectorCount":[i for i in range(30)],
    "GA Model":[GA_Counter[sector] for sector in range(30)],
    "Real World":[Real_Counter[sector] for sector in range(30)]}
SectorDataFrame=pd.DataFrame(data)
SectorDataFrame.plot.bar(x='SectorCount',figsize=(10,5),ylabel = 'Number of sectors')
plt.savefig(f"OutputImages/AirspaceTraffic_{TEST_DAY}.png")
TrafAvgReal=sum(RealTrafFactor.values())/1250
TrafAvgGA=sum(GATrafFactor.values())/1250
print(f"Average sector count in Real Data {TrafAvgReal}")
print(f"Average sector count in GA Solution {TrafAvgGA}")


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

plt.figure(5)
fig=pk.load(open("Notebooks/Outputs/Simulator.pkl","rb"))
plt.figure(figsize=(20,20))
realData=RealTrafFactor
hue=sorted([(k,v) for k,v in realData.items()],key=lambda x:x[1])
colors=getColours(NumColors)
seen=[False]*NumColors
for sector,k in hue:
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
plt.savefig(f"OutputImages/RealTraffic{TEST_DAY}.jpg")

plt.figure(6)
fig=pk.load(open("Notebooks/Outputs/Simulator.pkl","rb"))
plt.figure(figsize=(20,20))
gaDict=GATrafFactor
hue=sorted([(k,v) for k,v in gaDict.items()],key=lambda x:x[1])
seen=[False]*NumColors
for sector,k in hue:
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

plt.figure(figsize=(10,10))
AirportMatrix=Airportfile.read().splitlines()
GA_AirportOccupancy=dict()
AirportRunwayDict=dict()
NoZeroApOcGA=[]
NoZeroApOcReal=[]
for line in AirportMatrix:
    linelist=line.split(' ')
    Airport=linelist[0]
    NumRunways=int(linelist[1])
    AirportRunwayDict[Airport]=NumRunways
    linelist=linelist[2].split(',')
    linelist = [ int(x) for x in linelist ]
    noZeroList=[]
    for i in linelist:
        if(i>0):
            noZeroList.append(i)
    if(len(noZeroList)==0):
        GA_AirportOccupancy[Airport]=0
    else:
        GA_AirportOccupancy[Airport]=(max(noZeroList)/(NumRunways))*100
        NoZeroApOcGA.append(GA_AirportOccupancy[Airport])
realAirportOccupancy=dict()
for air in AirportTrafficReal:
    NumRunways=AirportRunwayDict[air]
    noZeroList=[]
    for i in AirportTrafficReal[air]:
        if(i>0):
            noZeroList.append(i)
    if(len(noZeroList)==0):
        realAirportOccupancy[air]=0
    else:
        realAirportOccupancy[air]=(max(noZeroList)/(NumRunways))*100
        NoZeroApOcReal.append(realAirportOccupancy[air])
plt.hist(NoZeroApOcReal, alpha=0.5, label='Real Airport Occupancy')
plt.hist(NoZeroApOcGA, alpha=0.5, label='GA Airport Occupancy')
plt.xlabel("Percentage Occupancy")
plt.ylabel("Number of Airports")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
plt.legend()
plt.savefig(f"OutputImages/AirportOccupancy{TEST_DAY}.jpg")
ApOcGA=0
ApOcReal=0
if(len(NoZeroApOcGA)>0):
    ApOcGA=sum(NoZeroApOcGA)/len(NoZeroApOcGA)
if(len(NoZeroApOcReal)>0):
    ApOcReal=sum(NoZeroApOcReal)/len(NoZeroApOcReal)
print(f"Average Airport Occupancy in Real Data {ApOcReal} %")
print(f"Average Airport Occupancy  in GA Solution {ApOcGA} %")
