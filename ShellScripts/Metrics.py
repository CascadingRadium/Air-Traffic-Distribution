from scipy.spatial import distance as dst
from mpl_toolkits.basemap import Basemap
from collections import defaultdict
import matplotlib.pyplot as plt
import shapely.geometry as sp
import datetime as dt
import pandas as pd
import pickle as pk
import numpy as np
import itertools
import math
import time
import json
import copy
import sys
TEST_DAYS=["2013-08-16"]
NumFlights=int(sys.argv[1])

cleanTracks=pd.read_csv("Outputs/cleanTracks.csv")
cleanFlights=pd.read_csv("Outputs/cleanFlightHistory.csv")
for date in TEST_DAYS:
	tracktemp=cleanTracks[cleanTracks['received'].str.contains(date)].copy()
	flighttemp=cleanFlights[cleanFlights['scheduled_runway_departure'].str.contains(date)].copy()
	tracktemp.to_csv(f"MetricFiles/DayWiseInput/{date}_tracks.csv",index=False)
	flighttemp.to_csv(f"MetricFiles/DayWiseInput/{date}_flights.csv",index=False)


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


