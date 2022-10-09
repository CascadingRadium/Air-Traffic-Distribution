#include <stdio.h>
#include <vector>
#include <fstream>
#include "cuda_runtime_api.h"
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#include <unistd.h>
#define NumThreads 32
#define MaxPathLen 1250
#define SectorTimeDictCols 2880
#define RAND_MAX 2147483647
#define MaxDelay 20
#define PI 3.141592653589793238
const double RadConvFactorToMultiply=180/PI;
#include "GeneticAlgorithm.h"
#include "HelperCFunctions.c"
int main()
{	
	/*Input File Name*/
	std::string InputFileName="InputFromFrontend.txt";
	/*Frontend Output File Name*/
	std::string OutputToFrontendFileName="OutputToFrontend.txt";
	/*Supplementary Files */
	std::string GraphFileName="CppGraph.txt";
	std::string GA_ParametersFileName="GA_Parameters.txt";
	/* GA Parameters*/
	int PopulationSize;
	int NumberOfMutations;
	int NumberOfGenerations;
	readGA_Params(PopulationSize,NumberOfMutations,NumberOfGenerations,GA_ParametersFileName);
	/* Read OD Pairs */
	std::vector<std::pair<Airport,Airport>> ODPairs;
	std::vector<int> times;
	std::vector<double> speed;
	int NumSectors=1250;
	readInput(ODPairs,InputFileName,times,speed);
	int NumODPairs=ODPairs.size();	
	std::vector<std::pair<std::vector<int>,PathOutput>>Paths(NumODPairs);
	/* Call CUDA Genetic Algorithm to solve the Congestion Game*/
	getPaths(ODPairs,Paths,NumSectors,PopulationSize,NumberOfMutations,NumberOfGenerations,GraphFileName,times,speed);// Input,Output
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess) 
		printf("CUDA error: %s\n",cudaGetErrorString(err)); 
	cudaProfilerStop();
	/*Output all Paths to Output File for the Frontend to read*/
	writeOutput(Paths,OutputToFrontendFileName,NumODPairs);
	return 0;
}
//void fitNessTest(double* device_Fitness,int PopulationSize)
//{
//	double maxx=0;
//	for(int i=0;i<(MaxDelay*PopulationSize);i++)
//	{
//		maxx=max(device_Fitness[i],maxx);
//	}
//	printf("%0.5g\n",maxx);
//		
//}
void getPaths(std::vector<std::pair<Airport,Airport>> &ODPairs, std::vector<std::pair<std::vector<int>,PathOutput>> &Paths, int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, std::string& GraphFileName, std::vector<int>&times, std::vector<double> &speeds)
{	
	GraphNode* host_graph[NumSectors];
	int host_arrSizes[NumSectors];
	int* SectorTimeDict; //2D
	int NumODPairs=ODPairs.size();
	AirportCoordinates* device_SourceCoordArr;
	AirportCoordinates* device_DestCoordArr;
	AirportCoordinates host_SourceCoordArr[NumODPairs];
	AirportCoordinates host_DestCoordArr[NumODPairs];
	GraphNode** device_graph;
	int *device_arrSizes;
	int* device_Paths; //2D
	int* device_Paths_size;
	double* device_Fitness; //2D
	int* SelectionPool;
	int* host_SelectionPool;
	int* Selected;
	int* SelectedDelay;
	int* device_times;
	//OUTPUT RELATED
	int* OutputPaths; //2D
	int* OutputPathsSizes;
	int* OutputPathsTime;
	int* OutputDelays;
	int* OutputAirTime;
	int* host_OutputPaths; //2D
	int* host_OutputPathsSizes;
	int* host_OutputDelays;
	int* host_OutputAirTime;
	int* TrafficMatrixSum;
	int SelectionSize=PopulationSize/2;
	if(SelectionSize&1==1)
		SelectionSize+=1;
	int CrossoverSize=SelectionSize/2;	
	for(int i=0;i<NumODPairs;i++)
	{
		host_SourceCoordArr[i].X=ODPairs[i].first.X;
		host_SourceCoordArr[i].Y=ODPairs[i].first.Y;
		host_DestCoordArr[i].X=ODPairs[i].second.X;
		host_DestCoordArr[i].Y=ODPairs[i].second.Y;
	}
	CUDA_Init(GraphFileName, host_graph, host_arrSizes, SectorTimeDict, device_SourceCoordArr, device_DestCoordArr, host_SourceCoordArr, host_DestCoordArr, device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, SelectionPool, host_SelectionPool, Selected, SelectedDelay, device_times,SelectionSize, NumSectors, PopulationSize, NumODPairs, OutputPaths, OutputPathsSizes, OutputDelays, host_OutputPaths, host_OutputPathsSizes, host_OutputDelays, OutputPathsTime, OutputAirTime, host_OutputAirTime, TrafficMatrixSum);
	for(int i=0;i<NumODPairs;i++)
	{	
		double MpM_Speed=(speeds[i]*30.8667);
		GeneticAlgorithm(NumSectors, PopulationSize, SelectionSize, CrossoverSize, NumberOfMutations, NumberOfGenerations, ODPairs[i].first.sector, ODPairs[i].second.sector, SectorTimeDict, device_SourceCoordArr, device_DestCoordArr, device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, SelectionPool, Selected, SelectedDelay, device_times, OutputPaths, OutputPathsSizes, OutputDelays, i, times[i], MpM_Speed, OutputPathsTime, OutputAirTime, TrafficMatrixSum);
		update_SectorTimeDict<<<(MaxPathLen/NumThreads)+1,NumThreads>>>(SectorTimeDict, OutputPaths, OutputDelays, OutputPathsSizes, i, times[i], OutputPathsTime, TrafficMatrixSum);
		resetForNextPair(device_Paths, device_times, device_Paths_size, Selected, SelectedDelay, PopulationSize, SelectionSize, OutputPathsTime);
	}
	cudaMemcpy(host_OutputPaths,OutputPaths,sizeof(int)*MaxPathLen*NumODPairs,cudaMemcpyDeviceToHost);
	cudaMemcpy(host_OutputPathsSizes,OutputPathsSizes,sizeof(int)*NumODPairs,cudaMemcpyDeviceToHost);
	cudaMemcpy(host_OutputDelays,OutputDelays,sizeof(int)*NumODPairs,cudaMemcpyDeviceToHost);
	cudaMemcpy(host_OutputAirTime,OutputAirTime,sizeof(int)*NumODPairs,cudaMemcpyDeviceToHost);
	for(int i=0;i<NumODPairs;i++)
	{
		int Loc=i*MaxPathLen;
		for(int j=0;j<host_OutputPathsSizes[i];j++)
		{
			Paths[i].first.push_back(host_OutputPaths[Loc+j]);
		}
		Paths[i].second.EstimatedDeparture=times[i];
		Paths[i].second.GroundHolding=host_OutputDelays[i];
		Paths[i].second.ActualDeparture=times[i]+host_OutputDelays[i];
		Paths[i].second.AerialDelay=host_OutputAirTime[i];
		Paths[i].second.ArrivalTime=Paths[i].second.ActualDeparture+Paths[i].second.AerialDelay;
		Paths[i].second.speed=speeds[i];
		Paths[i].second.StartICAO=ODPairs[i].first.ICAO;
		Paths[i].second.EndICAO=ODPairs[i].second.ICAO;
	}
	CUDA_Free(SectorTimeDict, device_SourceCoordArr, device_DestCoordArr, device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, SelectionPool, host_SelectionPool, Selected, SelectedDelay, device_times, OutputPaths, OutputPathsSizes, OutputPathsTime, OutputDelays, OutputAirTime, host_OutputPaths, host_OutputPathsSizes, host_OutputDelays, host_OutputAirTime,TrafficMatrixSum);
}
void CUDA_Init(std::string &GraphFileName, GraphNode** host_graph, int* host_arrSizes,int* &SectorTimeDict, AirportCoordinates* &device_SourceCoordArr, AirportCoordinates* &device_DestCoordArr, AirportCoordinates* host_SourceCoordArr, AirportCoordinates* host_DestCoordArr, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedDelay, int* &device_times ,int SelectionSize, int NumSectors, int PopulationSize, int NumODPairs, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputDelays, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &OutputPathsTime ,int* &OutputAirTime, int* &host_OutputAirTime,int* &TrafficMatrixSum)
{
	//ONE TIME
	srand(time(NULL));
	readGraph(GraphFileName,host_graph,host_arrSizes);
	gpuErrchk(cudaMalloc((void **)&device_arrSizes, sizeof(int)*NumSectors));	
	gpuErrchk(cudaMemcpy(device_arrSizes, host_arrSizes, sizeof(int)*(NumSectors),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void ***)&(device_graph), sizeof(GraphNode*)*NumSectors));
	for(int i=0;i<NumSectors;i++)
	{
		GraphNode* temp;
		cudaMalloc((void**)&temp, sizeof(GraphNode) * host_arrSizes[i]);
		cudaMemcpy(temp, host_graph[i], sizeof(GraphNode) * host_arrSizes[i], cudaMemcpyHostToDevice);
		cudaMemcpy(device_graph+i, &temp, sizeof(GraphNode*), cudaMemcpyHostToDevice);
	}
	gpuErrchk(cudaMalloc((void**)&SelectionPool, sizeof(int)*PopulationSize));
	host_SelectionPool=(int*)calloc(sizeof(int),PopulationSize);
	for(int i=0;i<PopulationSize;i++)
		host_SelectionPool[i]=i;
	gpuErrchk(cudaMemcpy(SelectionPool,host_SelectionPool,sizeof(int)*PopulationSize,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&SectorTimeDict,sizeof(int)*NumSectors*SectorTimeDictCols));
	gpuErrchk(cudaMemset(SectorTimeDict,0,sizeof(int)*NumSectors*SectorTimeDictCols));
	gpuErrchk(cudaMalloc((void**)&OutputPaths,sizeof(int)*MaxPathLen*NumODPairs));
	gpuErrchk(cudaMemset(OutputPaths,0,sizeof(int)*MaxPathLen*NumODPairs));
	gpuErrchk(cudaMalloc((void**)&OutputPathsSizes,sizeof(int)*NumODPairs));
	gpuErrchk(cudaMemset(OutputPathsSizes,0,sizeof(int)*NumODPairs));
	gpuErrchk(cudaMalloc((void**)&OutputDelays,sizeof(int)*NumODPairs));
	gpuErrchk(cudaMemset(OutputDelays,0,sizeof(int)*NumODPairs));
	gpuErrchk(cudaMalloc((void**)&OutputAirTime,sizeof(int)*NumODPairs));
	gpuErrchk(cudaMemset(OutputAirTime,0,sizeof(int)*NumODPairs));
	gpuErrchk(cudaMalloc((void **)&device_SourceCoordArr, sizeof(AirportCoordinates)*NumODPairs));
	gpuErrchk(cudaMemcpy(device_SourceCoordArr, host_SourceCoordArr, sizeof(AirportCoordinates)*NumODPairs,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void **)&device_DestCoordArr, sizeof(AirportCoordinates)*NumODPairs));
	gpuErrchk(cudaMemcpy(device_DestCoordArr, host_DestCoordArr, sizeof(AirportCoordinates)*NumODPairs,cudaMemcpyHostToDevice));
	
	host_OutputPaths=(int*)calloc(sizeof(int),NumODPairs*MaxPathLen);
	host_OutputPathsSizes=(int*)calloc(sizeof(int),NumODPairs);
	host_OutputDelays=(int*)calloc(sizeof(int),NumODPairs);
	host_OutputAirTime=(int*)calloc(sizeof(int),NumODPairs);
	gpuErrchk(cudaMalloc((void**)&TrafficMatrixSum,sizeof(int)));
	int tr[1]={1};
	gpuErrchk(cudaMemcpy(TrafficMatrixSum,tr,sizeof(int),cudaMemcpyHostToDevice));
	

	//RESET PER OD PAIR
	gpuErrchk(cudaMalloc((void **)&device_Fitness, sizeof(double)*PopulationSize*(MaxDelay))); //DO NOT MEMSET -> DOUBLE
	gpuErrchk(cudaMalloc((void **)&device_Paths, sizeof(int)*PopulationSize*MaxPathLen));
	gpuErrchk(cudaMemset(device_Paths,0,sizeof(int)*PopulationSize*MaxPathLen));
	gpuErrchk(cudaMalloc((void **)&device_times, sizeof(int)*PopulationSize*MaxPathLen));
	gpuErrchk(cudaMemset(device_times,0,sizeof(int)*PopulationSize*MaxPathLen));
	gpuErrchk(cudaMalloc((void **)&device_Paths_size, sizeof(int)*PopulationSize));
	gpuErrchk(cudaMemset(device_Paths_size,0,sizeof(int)*PopulationSize));
	gpuErrchk(cudaMalloc((void**)&Selected, sizeof(int)*SelectionSize));
	gpuErrchk(cudaMemset(Selected,0,sizeof(int)*SelectionSize));
	gpuErrchk(cudaMalloc((void**)&SelectedDelay, sizeof(int)*SelectionSize));
	gpuErrchk(cudaMemset(SelectedDelay,0,sizeof(int)*SelectionSize));
	gpuErrchk(cudaMalloc((void**)&OutputPathsTime,sizeof(int)*MaxPathLen));
	gpuErrchk(cudaMemset(OutputPathsTime,0,sizeof(int)*MaxPathLen));
}
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize, int seed, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime, double speed, int* device_times, int* TrafficMatrixSum,int Index)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<PopulationSize)
	{	
		getPath(device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, PopulationSize, seed, device_SourceCoord, device_DestCoord, SectorTimeDict, start, end, thread, 0, StartTime, speed, device_times, TrafficMatrixSum,Index);
	}
}
void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int CrossoverSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, AirportCoordinates* &device_SourceCoord, AirportCoordinates* &device_DestCoord,  GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* & device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &Selected, int* &SelectedDelay, int* device_times, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int OutputIndex, int StartTime, double speed, int* &OutputPathsTime, int* &OutputAirTime, int* &TrafficMatrixSum)
{	
	getInitPopulation<<<(PopulationSize/NumThreads)+1,NumThreads>>> (device_graph,device_arrSizes,device_Paths,device_Paths_size,device_Fitness,Start,End,PopulationSize,time(NULL),device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,OutputIndex);
	int genNum=0;
	int SelectionPoolSize=PopulationSize;
	int OriginalNumberOfMutations=NumberOfMutations;
	while(genNum<NumberOfGenerations)
	{
		genNum+=1;
		if(NumberOfMutations==0)
			NumberOfMutations=OriginalNumberOfMutations;
		else
			NumberOfMutations=abs(rand())%NumberOfMutations;
		Shuffle<<<1,1>>>(SelectionPool,SelectionPoolSize,time(NULL));
		SelectionKernel<<<(SelectionSize/NumThreads)+1,NumThreads>>>(Selected,SelectionPool,device_Fitness,SelectionSize,SelectedDelay,PopulationSize);
		CrossoverShuffle<<<1,1>>>(Selected,SelectedDelay,SelectionSize,time(NULL));
		CrossoverKernel<<<(CrossoverSize/NumThreads)+1,NumThreads>>> (Selected,SelectedDelay,device_Paths,device_Paths_size,device_Fitness,CrossoverSize,time(NULL),device_graph, device_arrSizes,device_times,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,TrafficMatrixSum,OutputIndex);
		if(NumberOfMutations!=0)
		{
			Shuffle<<<1,1>>>(SelectionPool,SelectionPoolSize,time(NULL));
			Mutation<<<(NumberOfMutations/NumThreads)+1,NumThreads>>>(SelectionPool, NumberOfMutations, time(NULL), device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, PopulationSize, device_SourceCoord, device_DestCoord, SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,OutputIndex);
		}
		Repair<<<(PopulationSize/NumThreads)+1,NumThreads>>>(device_Paths, device_Paths_size, PopulationSize, device_Fitness, device_graph, device_arrSizes, device_SourceCoord, device_DestCoord,  SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,OutputIndex);
	}
	getOutput<<<1,1>>>(device_Fitness, device_Paths, device_Paths_size, PopulationSize, OutputPaths, OutputPathsSizes, OutputDelays, OutputIndex, OutputPathsTime, device_graph, device_arrSizes, speed, device_times, StartTime, OutputAirTime, device_SourceCoord, device_DestCoord); 
}
void resetForNextPair(int* &device_Paths, int* &device_times, int* &device_Paths_size, int* &Selected, int* &SelectedDelay, int PopulationSize, int SelectionSize, int* &OutputPathsTime)
{
	gpuErrchk(cudaMemset(device_Paths,0,sizeof(int)*PopulationSize*MaxPathLen));
	gpuErrchk(cudaMemset(device_times,0,sizeof(int)*PopulationSize*MaxPathLen))
	gpuErrchk(cudaMemset(device_Paths_size,0,sizeof(int)*PopulationSize));
	gpuErrchk(cudaMemset(Selected,0,sizeof(int)*SelectionSize));
	gpuErrchk(cudaMemset(SelectedDelay,0,sizeof(int)*SelectionSize));
	gpuErrchk(cudaMemset(OutputPathsTime,0,sizeof(int)*MaxPathLen));
}
void CUDA_Free(int* &SectorTimeDict, AirportCoordinates* &device_SourceCoordArr, AirportCoordinates* &device_DestCoordArr, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedDelay, int* &device_times, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputPathsTime, int* &OutputDelays, int* &OutputAirTime, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &host_OutputAirTime,int* &TrafficMatrixSum)
{
	cudaFree(SectorTimeDict);
	cudaFree(device_SourceCoordArr);
	cudaFree(device_DestCoordArr);
	cudaFree(device_graph);
	cudaFree(device_arrSizes);
	cudaFree(device_Paths);
	cudaFree(device_Paths_size);
	cudaFree(device_Fitness);
	cudaFree(SelectionPool);
	cudaFree(Selected);
	cudaFree(SelectedDelay);
	cudaFree(device_times);
	cudaFree(OutputPaths);
	cudaFree(OutputPathsSizes);
	cudaFree(OutputPathsTime);
	cudaFree(OutputDelays);
	cudaFree(OutputAirTime);
	cudaFree(TrafficMatrixSum);
	free(host_SelectionPool);
	free(host_OutputPaths);
	free(host_OutputPathsSizes);
	free(host_OutputDelays);
	free(host_OutputAirTime);
	cudaDeviceReset();
}
__device__ void getPath(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, int seed, AirportCoordinates* &device_SourceCoord, AirportCoordinates* &device_DestCoord, int* SectorTimeDict, int start, int end, int thread, int skip, int StartTime, double speed, int* device_times,int* &TrafficMatrixSum,int Index)
{
	curandState_t state;
	curand_init(seed, thread, 0, &state);
	bool visited[MaxPathLen];
	int ptr_pos=skip;
	int Loc=thread*MaxPathLen;
	device_Paths[Loc+ptr_pos++]=start;
	bool InitPath=false;
	int validIndex[20];
	int validIndexSize=0;
	int num_neighbors;
	int cur;
	while(!InitPath)
	{
		memset(visited,0,MaxPathLen);
		if(skip!=0)
		{
			for(int i=0;i<skip;i++)
				visited[device_Paths[Loc+i]]=true;
		}
		visited[start]=true;
		ptr_pos=skip;
		device_Paths[Loc+ptr_pos++]=start;
		cur=start;
		while(!InitPath)
		{
			validIndexSize=0;
			num_neighbors=device_arrSizes[cur];
			for(int i=0;i<num_neighbors;i++)
			{
				if(!visited[device_graph[cur][i].vertexID])
					validIndex[validIndexSize++]=i;
			}
			if(validIndexSize==0)
				break;
			else
			{
				cur=device_graph[cur][validIndex[abs((int)curand(&state))%validIndexSize]].vertexID;
				visited[cur]=true;
				device_Paths[Loc+ptr_pos++]=cur;
				if(cur==end)
					InitPath=true;
			}
		}
	}
	device_Paths_size[thread]=ptr_pos;
	InitPathFitness(device_Fitness,device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index);
}
__device__ double getAngle(double Ax,double Ay, double Bx, double By, double Cx, double Cy)
{
	double a = atan2(-(By-Ay),Bx-Ax)*RadConvFactorToMultiply;
	double b = atan2(-(By-Cy),Bx-Cx)*RadConvFactorToMultiply;
	if(abs(b-a)>180)
		return 180-(360-abs(b-a));
	else
		return 180-(abs(b-a));
}
__device__ double euclidianDistance(double Point1X, double Point1Y, double Point2X, double Point2Y)
{
	double sumsq = 0.0;
	sumsq += pow(Point2X - Point1X, 2);
	sumsq += pow(Point2Y - Point1Y, 2);
	return sqrt(sumsq);
}
__device__ void InitPathFitness(double* device_Fitness, int* device_Paths, int* device_Paths_size, int thread, GraphNode** device_graph, int* device_arrSizes, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime, double speed, int* device_times,int* TrafficMatrixSum, int AirportIndex)
{
	int i=0;
	int j=0;
	int FitLoc=thread*MaxDelay;
	int Loc=thread*MaxPathLen;
	int CurSec=device_Paths[Loc];
	int NextSec=device_Paths[Loc+1];
	double prevPointX=(device_SourceCoord+AirportIndex)->X;
	double prevPointY=(device_SourceCoord+AirportIndex)->Y;
	double curPointX=0;
	double curPointY=0;
	double AnglePointsX[MaxPathLen+1];
	double AnglePointsY[MaxPathLen+1];
	double dist=0;
	int Index=0;
	AnglePointsX[Index]=prevPointX;
	AnglePointsY[Index++]=prevPointY;
	double path_length=0;
	double angle = 1;
	double TrafficFactor=1;
	int delay=0;
	double StaticCost=0;
	for (i=0;i<device_Paths_size[thread]-1;i++)
	{
		CurSec=device_Paths[Loc+i];
		NextSec=device_Paths[Loc+(i+1)];
		for(int j=0;j<device_arrSizes[CurSec];j++)
		{
			if(device_graph[CurSec][j].vertexID==NextSec)
			{
				curPointX=device_graph[CurSec][j].XCoord;
				curPointY=device_graph[CurSec][j].YCoord;
				dist=euclidianDistance(prevPointX,prevPointY,curPointX,curPointY);
				device_times[Loc+i]=ceil(dist/speed);
				path_length+=dist;
				AnglePointsX[Index]=curPointX;
				AnglePointsY[Index++]=curPointY;
				prevPointX=curPointX;
				prevPointY=curPointY;
				break;
			}
		}
	}
	dist=euclidianDistance(prevPointX,prevPointY,(device_DestCoord+AirportIndex)->X,(device_DestCoord+AirportIndex)->Y);
	device_times[Loc+i]=ceil(dist/speed);
	path_length+=dist;
	AnglePointsX[Index]=(device_DestCoord+AirportIndex)->X;
	AnglePointsY[Index++]=(device_DestCoord+AirportIndex)->Y;
	for (i=0;i<Index-2;i++)
	{
		angle+=getAngle(AnglePointsX[i],AnglePointsY[i],AnglePointsX[i+1],AnglePointsY[i+1],AnglePointsX[i+2],AnglePointsY[i+2]);
	}
	StaticCost=(path_length*angle);
	int InnerLoc=(device_Paths[Loc]*MaxPathLen);
	for(delay=0;delay<MaxDelay;delay++)
	{
		TrafficFactor=0;
		double time=device_times[Loc]+StartTime+delay;
		for(j=StartTime+delay;j<time;j++)
			TrafficFactor+=SectorTimeDict[InnerLoc+j];	
		for(i=1;i<device_Paths_size[thread];i++)
		{
			InnerLoc=(device_Paths[Loc+i]*MaxPathLen);	
			for(j=time;j<time+device_times[Loc+i];j++)
			{
				TrafficFactor+=SectorTimeDict[InnerLoc+j];	
			}
			time=time+device_times[Loc+i];
		}
		device_Fitness[FitLoc+delay]=(double)((MaxDelay-delay)+(*(TrafficMatrixSum)-TrafficFactor))/(StaticCost);
	}
}
__global__ void update_SectorTimeDict(int* SectorTimeDict, int* OutputPaths, int* OutputDelays, int* OutputPathsSize, int Index, int StartTime, int* OutputPathsTime, int* TrafficMatrixSum)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	int PathLen=OutputPathsSize[Index];
	if(thread != 0 && thread < PathLen)
	{
		int Loc=(OutputPaths[Index*MaxPathLen+thread]*MaxPathLen);
		int Sum=0;
		for(int i=OutputPathsTime[thread-1];i<OutputPathsTime[thread];i++)
		{
			SectorTimeDict[Loc+i]+=1;
			Sum+=1;
		}
		atomicAdd(TrafficMatrixSum, Sum);
	}
	else if(thread==0)
	{
		int Loc=(OutputPaths[Index*MaxPathLen]*MaxPathLen);
		int Sum=0;
		for(int i=StartTime+OutputDelays[Index];i<OutputPathsTime[0];i++)
		{
			SectorTimeDict[Loc+i]+=1;
			Sum+=1;
		}
		atomicAdd(TrafficMatrixSum, Sum);
	}
}
__global__ void Shuffle(int* SelectionPool,int SelectionPoolSize,int seed)
{
	curandState_t state;
	curand_init(seed, 0, 0, &state);
	for (int i = 0; i < SelectionPoolSize - 1; i++) 
	{
		int j = i + abs((int)curand(&state)) / (RAND_MAX / (SelectionPoolSize - i) + 1);
		int t = SelectionPool[j];
		SelectionPool[j] = SelectionPool[i];
		SelectionPool[i] = t;
	}
}
__global__ void CrossoverShuffle(int* Selection,int* SelectedDelay,int SelectionSize,int seed)
{
	curandState_t state;
	curand_init(seed, 0, 0, &state);
	for (int i = 0; i < SelectionSize - 1; i++) 
	{
		int j = i + abs((int)curand(&state)) / (RAND_MAX / (SelectionSize - i) + 1);
		int c = Selection[j];
		int t = SelectedDelay[j];
		Selection[j] = Selection[i];
		Selection[i] = c;
		SelectedDelay[j]=SelectedDelay[i];
		SelectedDelay[i]=t;
	}
}
__global__ void SelectionKernel(int* Selected, int*SelectionPool, double* device_Fitness,int SelectionSize,int*SelectedDelay,int PopulationSize)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<SelectionSize)
	{
		double max1=0.0,max2=0.0;
		double avg1=0.0,avg2=0.0;
		int t1=-1,t2=-1;
		int Loc1=SelectionPool[2*thread]*MaxDelay;
		int Loc2=SelectionPool[2*thread+1]*MaxDelay;
		for(int i=0;i<MaxDelay;i++)
		{
			avg1+=device_Fitness[Loc1+i];
			if(device_Fitness[Loc1+i]>max1)
			{
				max1=device_Fitness[Loc1+i];
				t1=i;
			}
		}
		for(int i=0;i<MaxDelay;i++)
		{
			avg2+=device_Fitness[Loc2+i];
			if(device_Fitness[Loc2+i]>max2)
			{
				max2=device_Fitness[Loc2+i];
				t2=i;
			}
		}
		if(avg1<avg2)
		{
			Selected[thread]=SelectionPool[2*thread+1];
			SelectedDelay[thread]=t2;
		}
		else
		{
			Selected[thread]=SelectionPool[2*thread];
			SelectedDelay[thread]=t1;
		}
	}
}

__global__ void CrossoverKernel(int* Selected, int* SelectedDelay, int* device_Paths, int* device_Paths_size, double* device_Fitness, int CrossoverSize,int seed,GraphNode** device_graph, int* device_arrSizes,int* device_times, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict,int StartTime,double speed,int* TrafficMatrixSum,int Index)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<CrossoverSize)
	{	
		curandState_t state;
		curand_init(seed, thread, 0, &state);
		int CommonIndecesFirst[MaxPathLen]={-1};
		int CommonIndecesSecond[MaxPathLen]={-1};
		int CommonIndecesLen=0;
		int CumulativeTime1[MaxPathLen];
		int CumulativeTime2[MaxPathLen];
		int i=0;
		int j=0;
		int l1=0;
		int l2=0;
		int r1=0;
		int r2=0;
		int FirstIndex=2*thread;
		int SecondIndex=(2*thread)+1;
		int Loc1=Selected[FirstIndex]*MaxPathLen;
		int Loc2=Selected[SecondIndex]*MaxPathLen;
		int FitLoc1=Selected[FirstIndex]*MaxDelay;
		int FitLoc2=Selected[SecondIndex]*MaxDelay;
		int FirstLen=device_Paths_size[Selected[FirstIndex]];
		int SecondLen=device_Paths_size[Selected[SecondIndex]];
		double avgFirst=0,avgSecond=0;
		CumulativeTime1[0]=StartTime+SelectedDelay[FirstIndex]+device_times[Loc1];
		CumulativeTime2[0]=StartTime+SelectedDelay[SecondIndex]+device_times[Loc2];
		for(i=1;i<FirstLen;i++)
			CumulativeTime1[i]=CumulativeTime1[i-1]+device_times[Loc1+i];
		for(i=1;i<SecondLen;i++)
			CumulativeTime2[i]=CumulativeTime2[i-1]+device_times[Loc2+i];
		i=0;
		j=0;
		while(i < FirstLen && j < SecondLen)
		{
			if(i==0)
			{
				l1=0;
				r1=CumulativeTime1[i]-1;
			}
			else
			{
				l1=CumulativeTime1[i-1];
				r1=CumulativeTime1[i]-1;
			}
			if(j==0)
			{
				l2=0;
				r2=CumulativeTime2[j]-1;
			}
			else
			{
				l2=CumulativeTime2[j-1];
				r2=CumulativeTime2[j]-1;
			}
			if(max(l1,l2) <= min(r1,r2) && !(i==0 && j==0) && !(i==FirstLen-1 && j== SecondLen-1))
			{
				if(device_Paths[Loc1+i]==device_Paths[Loc2+j])
				{
					CommonIndecesFirst[CommonIndecesLen]=i;
					CommonIndecesSecond[CommonIndecesLen++]=j;
				}
			}
			if(CumulativeTime1[i] < CumulativeTime2[j])
				i++;
			else
				j++;
		}
		if(CommonIndecesLen!=0)
		{	
			int TempArray[MaxPathLen]; // For Storing First Chm Second Part (which gets overwritten)
			r1=0;//Serves as tempArrayLen
			r2=0;//Serves as checkpoint at which to swap
			int CrossIndex=abs((int)curand(&state))%CommonIndecesLen;
			l1=CommonIndecesFirst[CrossIndex];  //l1 is index for first and l2 for second
			l2=CommonIndecesSecond[CrossIndex];
			for(i=0;i<MaxDelay;i++)
			{
				avgFirst+=device_Fitness[FitLoc1+i];
				avgSecond+=device_Fitness[FitLoc2+i];
			}
			int FirstFitnessCopy[MaxDelay]={-1};//CumulativeTime1 is FirstTimeCopy and CommonIndecesFirst is FirstCopy and FirstLen is device_path[first] copy
			int SecondFitnessCopy[MaxDelay]={-1}; //CumulativeTime2 is SecondTimeCopy and CommonIndecesSecond is SecondCopy and SecondLen is device_path[second] copy
			for(i=0;i<MaxDelay;i++)
			{
				FirstFitnessCopy[i]=device_Fitness[FitLoc1+i];
				SecondFitnessCopy[i]=device_Fitness[FitLoc2+i];
			}	
			for(i=0;i<FirstLen;i++)
			{
				CommonIndecesFirst[i]=device_Paths[Loc1+i];
				CumulativeTime1[i]=device_times[Loc1+i];
			}
			for(i=0;i<SecondLen;i++)
			{
				CommonIndecesSecond[i]=device_Paths[Loc2+i];
				CumulativeTime2[i]=device_times[Loc2+i];
			}
			for(i=l1+1;i<FirstLen;i++)
				TempArray[r1++]=device_Paths[Loc1+i];
			r2=l1+1;
			for(i=l2+1;i<SecondLen;i++)
				device_Paths[Loc1+r2++]=device_Paths[Loc2+i];
			device_Paths_size[Selected[FirstIndex]]=r2; //Reuse r2 for same thing in Second
			r2=l2+1;			
			for(i=0;i<r1;i++)
				device_Paths[Loc2+r2++]=TempArray[i];
			device_Paths_size[Selected[SecondIndex]]=r2;
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,Selected[FirstIndex],device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index);
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,Selected[SecondIndex],device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index);
			for(i=0;i<MaxDelay;i++)
			{
				avgFirst-=device_Fitness[FitLoc1+i];
				avgSecond-=device_Fitness[FitLoc2+i];
			}
			if(avgFirst>0)
			{
				for(i=0;i<FirstLen;i++)
				{
					device_Paths[Loc1+i]=CommonIndecesFirst[i];
					device_times[Loc1+i]=CumulativeTime1[i];
				}
				device_Paths_size[Selected[FirstIndex]]=FirstLen;
				for(i=0;i<MaxDelay;i++)
					device_Fitness[FitLoc1+i]=FirstFitnessCopy[i];
			}
			if(avgSecond>0)
			{
				for(i=0;i<SecondLen;i++)
				{
					device_Paths[Loc2+i]=CommonIndecesSecond[i];
					device_times[Loc2+i]=CumulativeTime2[i];
				}
				device_Paths_size[Selected[SecondIndex]]=SecondLen;
				for(i=0;i<MaxDelay;i++)
					device_Fitness[FitLoc2+i]=SecondFitnessCopy[i];
			}
		}
	}
}
__global__ void Mutation(int* MutationPool, int NumberOfMutations, int seed, GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, AirportCoordinates* device_SourceCoord,  AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime,double speed,int* device_times,int* TrafficMatrixSum,int Index)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<NumberOfMutations)
	{
		curandState_t state;
		curand_init(seed, thread, 0, &state);
		int pathID=MutationPool[thread];
		int MutPoint=abs((int)curand(&state))%(device_Paths_size[pathID]-2);
		getPath(device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, PopulationSize, seed, device_SourceCoord, device_DestCoord, SectorTimeDict,device_Paths[pathID*MaxPathLen+MutPoint+1], device_Paths[pathID*MaxPathLen+(device_Paths_size[pathID]-1)], pathID, MutPoint+1, StartTime,speed,device_times,TrafficMatrixSum,Index);
	}
}
__global__ void Repair(int* device_Paths, int* device_Paths_size, int PopulationSize, double* device_Fitness, GraphNode** device_graph, int* device_arrSizes, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict,int StartTime,double speed,int* device_times,int* TrafficMatrixSum,int Index)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<PopulationSize)
	{
		int dict[1250]={-1};
		int Pos=thread*MaxPathLen;
		int i=0;
		for(i=0;i<device_Paths_size[thread];i++)
			dict[device_Paths[Pos+i]]=i;
		i=0;
		int index=0;
		while(index<device_Paths_size[thread])
		{
			device_Paths[Pos+i]=device_Paths[Pos+index];
			index=dict[device_Paths[Pos+index]]+1;
			++i;
		}
		if(i<device_Paths_size[thread])
		{
			device_Paths_size[thread]=i;
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index);
		}
	}
}
__global__ void getOutput(double* device_Fitness,int* device_Paths, int* device_Paths_size, int PopulationSize, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int index, int* OutputPathsTime, GraphNode** device_graph, int* device_arrSizes,double speed, int* device_times,int StartTime, int* OutputAirTime, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord)
{
	double maxF=device_Fitness[0];
	int path_index=0;
	int time=0;
	int i=1;
	int OutLoc=index*MaxPathLen;
	double distance=0;
	for(i=1;i<MaxDelay*PopulationSize;i++)
	{
		if(device_Fitness[i]>maxF)
		{
			maxF=device_Fitness[i];
			path_index=i/MaxDelay;
			time=i%MaxDelay;
		}
	}
	int Loc=path_index*MaxPathLen;
	OutputPathsSizes[index]=device_Paths_size[path_index];
	OutputDelays[index]=time;
	int CurSec=device_Paths[Loc];
	int NextSec=device_Paths[Loc+1];
	double prevPointX=device_SourceCoord->X;
	double prevPointY=device_SourceCoord->Y;	
	double curPointX=0;
	double curPointY=0;
	for (i=0;i<device_Paths_size[path_index]-1;i++)
	{
		CurSec=device_Paths[Loc+i];
		NextSec=device_Paths[Loc+(i+1)];
		for(int j=0;j<device_arrSizes[CurSec];j++)
		{
			if(device_graph[CurSec][j].vertexID==NextSec)
			{
				curPointX=device_graph[CurSec][j].XCoord;
				curPointY=device_graph[CurSec][j].YCoord;
				distance+=euclidianDistance(prevPointX,prevPointY,curPointX,curPointY);
				prevPointX=curPointX;
				prevPointY=curPointY;
				break;
			}
		}
	}
	distance+=euclidianDistance(prevPointX,prevPointY,device_DestCoord->X,device_DestCoord->Y);
	OutputAirTime[index]=ceil(distance/speed);
	OutputPathsTime[0]=StartTime+time+device_times[Loc];
	OutputPaths[OutLoc]=device_Paths[Loc];
	for(i=1;i<device_Paths_size[path_index];i++)
	{
		OutputPaths[OutLoc+i]=device_Paths[Loc+i];
		OutputPathsTime[i]=device_times[Loc+i]+OutputPathsTime[i-1];
	}
}
