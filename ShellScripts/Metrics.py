from mpl_toolkits.basemap import Basemap
import shapely.geometry as sp
import datetime as dt
import pandas as pd
import pickle as pk
import math
import json
import copy
import sys
TEST_DAY=sys.argv[1]
NumFlights=int(sys.argv[2])

if(NumFlights!=-1):
    ConvexHulls = pk.load(open("../Notebooks/Outputs/ConvexDict.pkl", "rb"))
    m = pk.load(open("../Notebooks/Outputs/M_ConversionMetric.pkl", "rb"))
    SectorChunkDict=pk.load(open("../Notebooks/Outputs/SectorChunkDict.pkl", "rb"))
    AirportSectorDict=pk.load(open("../Notebooks/Outputs/airportSectorDict.pkl", "rb"))
    AirportDB=pk.load(open('../Notebooks/Outputs/icaoToNameDict.pkl', "rb"))
    speedDict=pk.load(open("../Notebooks/Outputs/SpeedDict.pkl",'rb'))

    dayTracks=pd.read_csv(f"../Dataset/{TEST_DAY}/{TEST_DAY}_tracks.csv")
    dayFlights=pd.read_csv(f"../Dataset/{TEST_DAY}/{TEST_DAY}_flights.csv")
    if(NumFlights!=-1):
        dayFlights=dayFlights[:NumFlights]
    dayTracks.sort_values(by=['flighthistory_id','received'],inplace=True,ignore_index=True)

    def getChunk(Point,SectorChunkDict):
        innerDict=SectorChunkDict[math.ceil(Point[0]/1000000)*1000000]
        Y=0
        for key in innerDict.keys():
            if key >= Point[1]:
                Y=key
                break
        return innerDict[Y] if Y!=0 else []
    def findSectorPath(path,sectorPath,z,SectorChunkDict,output):
        sector=-1
        for pointIdx in range(len(path)):
            sectorList=getChunk(path[pointIdx],SectorChunkDict)
            for i in sectorList:
                if len(output[i]) < 3:
                    continue
                if sp.Polygon(output[i]).contains(sp.Point(path[pointIdx])):
                    sector=i
                    break
            if sector!=-1:
                sectorPath.append(sector)
            else:
                z.append(pointIdx)

    NumSectors=1250
    TimeUnits=2880
    SectorTimeDict=[[0 for colnum in range(TimeUnits)] for rownum in range(NumSectors)]
    east,north=m(dayTracks["longitude_degrees"],dayTracks["latitude_degrees"])
    dayTracks["easting"]=east
    dayTracks["northing"]=north
    InvalidID=[]
    flightIDs=list(dayFlights["id"])
    for ID in flightIDs:
        sectorPath=[]
        z=[]
        tracks=dayTracks[dayTracks["flighthistory_id"]==ID].copy()
        tracks.reset_index(drop=True, inplace=True)
        if(len(tracks)==0):
            InvalidID.append(ID)
        else:
            path=list(zip(tracks["easting"],tracks["northing"]))
            sectorPath=[]
            z=[]
            findSectorPath(path,sectorPath,z,SectorChunkDict,ConvexHulls)
            tracks.drop(z, axis=0, inplace=True)
            timings=list(map(lambda x:dt.datetime.fromisoformat(str(x)),tracks["received"]))
            rowIdx=0
            NumRows=len(sectorPath)
            Flight=dayFlights[dayFlights["id"]==ID]
            startSec=AirportSectorDict[Flight["departure_airport_icao_code"].values[0]]
            endSec=AirportSectorDict[Flight["arrival_airport_icao_code"].values[0]]
            realSecPath=[]
            startTime=dt.datetime.fromisoformat(str(Flight["actual_runway_departure"].values[0]))
            endTime=dt.datetime.fromisoformat(str(Flight["actual_runway_arrival"].values[0]))
            realRowArr=[(startTime.hour*60)+(startTime.minute)]
            realSecArr=[startSec]
            seenSec=startSec
            row=0
            for sec in sectorPath:
                if(sec!=seenSec):
                    realRowArr.append((timings[row].hour*60)+timings[row].minute)
                    realSecArr.append(sec)
                    seenSec=sec
                if(sec==endSec):
                    break
                row+=1
            realRowArr.append((endTime.hour*60)+endTime.minute)
            for i in range(len(realSecArr)):
                for j in range(realRowArr[i],realRowArr[i+1]):
                    SectorTimeDict[realSecArr[i]][j]+=1
    kFile=open(f"RealMetrics/{TEST_DAY}_SectorTimeDict.pkl","wb")
    pk.dump(SectorTimeDict,kFile)
    kFile.close()
    K=dict()
    for i in range(NumSectors):
        K[i]=max(SectorTimeDict[i])
    kFile=open(f"RealMetrics/{TEST_DAY}_KDict.pkl","wb")
    pk.dump(K,kFile)
    kFile.close()

    ground_delay = []
    actual_air_time = []
    srd=list(map(lambda x:dt.datetime.fromisoformat(str(x)),dayFlights["scheduled_runway_departure"]))
    ard=list(map(lambda x:dt.datetime.fromisoformat(str(x)),dayFlights["actual_runway_departure"]))
    ara=list(map(lambda x:dt.datetime.fromisoformat(str(x)),dayFlights["actual_runway_arrival"]))
    NumRows=len(ard)
    for i in range(NumRows):
        ground_delay.append((ard[i] - srd[i])/pd.Timedelta(minutes=1))
        actual_air_time.append((ara[i]-ard[i])/pd.Timedelta(minutes=1))
    dayno_data = pd.DataFrame()
    dayno_data.insert(0, "Aerial Time", actual_air_time)
    dayno_data.insert(1, "ground_delay", ground_delay)
    dayno_data.to_csv(f"RealMetrics/{TEST_DAY}-gd-aat.csv",index=False)

    ids=list(dayFlights['id'])
    source=list(dayFlights["departure_airport_icao_code"])
    destination=list(dayFlights["arrival_airport_icao_code"])
    startTime=[dt.datetime.fromisoformat(str(date)) for date in dayFlights["scheduled_runway_departure"]]
    triplets=list(zip(ids,source,destination))
    start=[]
    for s in startTime:
        hour=s.hour
        minute=s.minute
        if minute < 10:
            minute="0" + str(minute)
        start.append(str(hour) + ":" + str(minute))
    toPrint=""
    for i in range(len(triplets)):
        flightID,sourceAirport,destinationAirport=triplets[i]
        sourceAirportName=AirportDB[sourceAirport]
        destinationAirportName=AirportDB[destinationAirport]
        if flightID not in speedDict:
            speedDict[flightID]=407.5
        toPrint+=f"{sourceAirportName},{destinationAirportName},1,{start[i]},{speedDict[flightID]}"
        if i!=len(triplets)-1:
            toPrint+="\n"
    f = open(f"{TEST_DAY}-WebsiteInput.txt", "w")
    f.write(toPrint)
    f.close()


    final_df = dayFlights.sort_values(by=['actual_runway_departure'], ascending=True,ignore_index=True)
    min_date=dt.datetime.fromisoformat(final_df['actual_runway_departure'][0])
    min_timestamp = int(round(min_date.timestamp()))
    def add_idif(final_df,name,min_timestamp):
        dtObjectSeries = []
        actualDep=final_df[name]
        for dep in actualDep:
            dtObjectSeries.append(dt.datetime.fromisoformat(dep))
        idifSeries = []
        for dep in dtObjectSeries:
            idif = int((int(round(dep.timestamp())) - min_timestamp)/60)
            idifSeries.append(idif)
        final_df[f'idif_{name}'] = idifSeries
    add_idif(final_df,'actual_runway_departure',min_timestamp)
    add_idif(final_df,'actual_runway_arrival',min_timestamp)

    RunwayFile=open("../Website/backend/InputFolder/AirportRunways.txt")

    Runways=RunwayFile.read().splitlines()
    AirportRunways=dict()
    for air in Runways:
        ls=air.split(',')
        AirportRunways[ls[0]]=int(ls[1])
    AirportOccupancy=dict()
    for k,v in AirportRunways.items():
        AirportOccupancy[k]= [0] * 2880
    final_df = dayFlights.sort_values(by=['actual_runway_departure'], ascending=True,ignore_index=True)
    min_date=dt.datetime.fromisoformat(final_df['actual_runway_departure'][0])
    min_timestamp = int(round(min_date.timestamp()))
    def add_idif(final_df,name,min_timestamp):
        dtObjectSeries = []
        actualDep=final_df[name]
        for dep in actualDep:
            dtObjectSeries.append(dt.datetime.fromisoformat(dep))
        idifSeries = []
        for dep in dtObjectSeries:
            idif = int((int(round(dep.timestamp())) - min_timestamp)/60)
            idifSeries.append(idif)
        final_df[f'idif_{name}'] = idifSeries
    add_idif(final_df,'actual_runway_departure',min_timestamp)
    add_idif(final_df,'actual_runway_arrival',min_timestamp)
    srcList=list(final_df["departure_airport_icao_code"])
    dstList=list(final_df["arrival_airport_icao_code"])
    stTime=list(final_df["idif_actual_runway_departure"])
    enTime=list(final_df["idif_actual_runway_arrival"])
    for flight in range(len(srcList)):
        AirportOccupancy[srcList[flight]][stTime[flight]]+=1
        AirportOccupancy[dstList[flight]][enTime[flight]]+=1
    AirportTrafficDict=open(f"RealMetrics/{TEST_DAY}_AirportTrafficDict.pkl","wb")
    pk.dump(AirportOccupancy,AirportTrafficDict)
    AirportTrafficDict.close()
