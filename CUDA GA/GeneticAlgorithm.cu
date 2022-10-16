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
#define ConvergenceCutoff 10
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
	/*Metric Files*/
	std::string TrafficFactorMetricFileName="GA_Metrics/TrafficFactor.txt";
	std::string AerGDFileName="GA_Metrics/AerialTimeGD.txt";
	/* Read OD Pairs */
	std::vector<std::pair<Airport,Airport>> ODPairs;
	std::vector<int> times;
	std::vector<double> speed;
	int NumSectors=1250;
	readInput(ODPairs,InputFileName,times,speed);
	int NumODPairs=ODPairs.size();	
	std::vector<std::pair<std::vector<int>,PathOutput>>Paths(NumODPairs);
	std::vector<int>TrafficFactorMetric;
	/* Call CUDA Genetic Algorithm to solve the Congestion Game*/
	getPaths(ODPairs,Paths,NumSectors,PopulationSize,NumberOfMutations,NumberOfGenerations,GraphFileName,times,speed,TrafficFactorMetric);// Input,Output
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess) 
		printf("CUDA error: %s\n",cudaGetErrorString(err)); 
	cudaProfilerStop();
	/*Output all Paths to Output File for the Frontend to read*/
	writeOutput(Paths,OutputToFrontendFileName,TrafficFactorMetric,TrafficFactorMetricFileName,AerGDFileName,NumODPairs);
	return 0;
}
void ConvergenceTest(double* device_FitnessArray,int NumRowsForPathMatrix,double &prevFitnessForConvergence, int& numTimesPrevFitnessOccured)
{
	double maxx=0;
	for(int i=0;i<(NumRowsForPathMatrix);i++)
	{
		maxx=max(device_FitnessArray[i],maxx);
	}
	if(prevFitnessForConvergence==maxx)
		numTimesPrevFitnessOccured++;
	else
	{
		prevFitnessForConvergence=maxx;
		numTimesPrevFitnessOccured=1;
	}
}
__global__ void copySecTimeDict(int* SectorTimeDict,int* metricSectorTimeDict)
{
	int thread = threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<MaxPathLen)
	{	
		int maxx=0;
		int Loc=thread*SectorTimeDictCols;
		for(int i=0;i<SectorTimeDictCols;i++)
			maxx=max(maxx,SectorTimeDict[Loc+i]);
		metricSectorTimeDict[thread]=maxx;
	}
}
void getPaths(std::vector<std::pair<Airport,Airport>> &ODPairs, std::vector<std::pair<std::vector<int>,PathOutput>> &Paths, int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, std::string& GraphFileName, std::vector<int>&times, std::vector<double> &speeds, std::vector<int> &TrafficFactorMetric)
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
	int* SelectionPool;
	int* host_SelectionPool;
	int* ReplacementPool;
	int* host_ReplacementPool;
	int* MutationPool;
	int* host_MutationPool;
	int* Selected;
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
	double* device_FitnessArray;
	int* device_TimeArray;
	Pair* heap;
	int SelectionSize=PopulationSize/2;
	int NumRowsForPathMatrix=(3*PopulationSize)/2;
	if(SelectionSize&1==1)
		SelectionSize+=1;
	int CrossoverSize=SelectionSize/2;
	int* metricSectorTimeDict;	
	int* host_metricSectorTimeDict;
	for(int i=0;i<NumODPairs;i++)
	{
		host_SourceCoordArr[i].X=ODPairs[i].first.X;
		host_SourceCoordArr[i].Y=ODPairs[i].first.Y;
		host_DestCoordArr[i].X=ODPairs[i].second.X;
		host_DestCoordArr[i].Y=ODPairs[i].second.Y;
	}
	CUDA_Init(GraphFileName, host_graph, host_arrSizes, SectorTimeDict, device_SourceCoordArr, device_DestCoordArr, host_SourceCoordArr, host_DestCoordArr, device_graph, device_arrSizes, device_Paths, device_Paths_size, SelectionPool, host_SelectionPool, Selected, device_times, SelectionSize, NumSectors, PopulationSize, NumODPairs, OutputPaths, OutputPathsSizes, OutputDelays, host_OutputPaths, host_OutputPathsSizes, host_OutputDelays, OutputPathsTime, OutputAirTime, host_OutputAirTime, TrafficMatrixSum, device_FitnessArray, device_TimeArray,NumRowsForPathMatrix,ReplacementPool,host_ReplacementPool,MutationPool,host_MutationPool,heap,metricSectorTimeDict,host_metricSectorTimeDict);
	for(int i=0;i<NumODPairs;i++)
	{	
		double MpM_Speed=(speeds[i]*30.8667);
		GeneticAlgorithm(NumSectors, PopulationSize, SelectionSize, CrossoverSize, NumberOfMutations, NumberOfGenerations, ODPairs[i].first.sector, ODPairs[i].second.sector, SectorTimeDict, device_SourceCoordArr, device_DestCoordArr, device_graph, device_arrSizes, device_Paths, device_Paths_size, SelectionPool, Selected, device_times, OutputPaths, OutputPathsSizes, OutputDelays, i, times[i], MpM_Speed, OutputPathsTime, OutputAirTime, TrafficMatrixSum,device_FitnessArray,device_TimeArray,NumRowsForPathMatrix,ReplacementPool,MutationPool,heap);
		update_SectorTimeDict<<<(MaxPathLen/NumThreads)+1,NumThreads>>>(SectorTimeDict, OutputPaths, OutputDelays, OutputPathsSizes, i, times[i], OutputPathsTime, TrafficMatrixSum);
		resetForNextPair(device_Paths, device_times, device_Paths_size, Selected, PopulationSize, SelectionSize, OutputPathsTime,device_FitnessArray,device_TimeArray);
	}
	copySecTimeDict<<<(MaxPathLen/NumThreads)+1,NumThreads>>>(SectorTimeDict,metricSectorTimeDict);
	cudaMemcpy(host_metricSectorTimeDict,metricSectorTimeDict,sizeof(int)*MaxPathLen,cudaMemcpyDeviceToHost);
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
	for(int i=0;i<MaxPathLen;i++)
		TrafficFactorMetric.push_back(host_metricSectorTimeDict[i]);
	CUDA_Free(SectorTimeDict, device_SourceCoordArr, device_DestCoordArr, device_graph, device_arrSizes, device_Paths, device_Paths_size, SelectionPool, host_SelectionPool, Selected, device_times, OutputPaths, OutputPathsSizes, OutputPathsTime, OutputDelays, OutputAirTime, host_OutputPaths, host_OutputPathsSizes, host_OutputDelays, host_OutputAirTime,TrafficMatrixSum,device_FitnessArray,device_TimeArray);
}
void CUDA_Init(std::string &GraphFileName, GraphNode** host_graph, int* host_arrSizes,int* &SectorTimeDict, AirportCoordinates* &device_SourceCoordArr, AirportCoordinates* &device_DestCoordArr, AirportCoordinates* host_SourceCoordArr, AirportCoordinates* host_DestCoordArr, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &device_times ,int SelectionSize, int NumSectors, int PopulationSize, int NumODPairs, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputDelays, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &OutputPathsTime ,int* &OutputAirTime, int* &host_OutputAirTime,int* &TrafficMatrixSum, double* &device_FitnessArray,int* &device_TimeArray,int NumRowsForPathMatrix,int* &ReplacementPool,int* &host_ReplacementPool,int* &MutationPool,int* &host_MutationPool, Pair* &heap, int* &metricSectorTimeDict,int* &host_metricSectorTimeDict)
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
	heap = (Pair*)malloc(sizeof(Pair)*NumRowsForPathMatrix);
	gpuErrchk(cudaMalloc((void**)&TrafficMatrixSum,sizeof(int)));
	int tr[1]={1};
	gpuErrchk(cudaMemcpy(TrafficMatrixSum,tr,sizeof(int),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&MutationPool, sizeof(int)*NumRowsForPathMatrix));
	
	gpuErrchk(cudaMalloc((void**)&metricSectorTimeDict,sizeof(int)*MaxPathLen));
	gpuErrchk(cudaMemset(metricSectorTimeDict,0,sizeof(int)*MaxPathLen));
	host_metricSectorTimeDict=(int*)calloc(sizeof(int),MaxPathLen);
	
	host_MutationPool=(int*)calloc(sizeof(int),NumRowsForPathMatrix);
	for(int i=0;i<NumRowsForPathMatrix;i++)
		host_MutationPool[i]=i;
	gpuErrchk(cudaMemcpy(MutationPool,host_MutationPool,sizeof(int)*NumRowsForPathMatrix,cudaMemcpyHostToDevice));

	//RESET PER OD PAIR
	gpuErrchk(cudaMalloc((void **)&device_Paths, sizeof(int)*NumRowsForPathMatrix*MaxPathLen));
	gpuErrchk(cudaMemset(device_Paths,0,sizeof(int)*NumRowsForPathMatrix*MaxPathLen));
	gpuErrchk(cudaMalloc((void **)&device_times, sizeof(int)*NumRowsForPathMatrix*MaxPathLen));
	gpuErrchk(cudaMemset(device_times,0,sizeof(int)*NumRowsForPathMatrix*MaxPathLen));
	gpuErrchk(cudaMalloc((void **)&device_Paths_size, sizeof(int)*NumRowsForPathMatrix));
	gpuErrchk(cudaMemset(device_Paths_size,0,sizeof(int)*NumRowsForPathMatrix));
	gpuErrchk(cudaMalloc((void**)&Selected, sizeof(int)*SelectionSize));
	gpuErrchk(cudaMemset(Selected,0,sizeof(int)*SelectionSize));
	gpuErrchk(cudaMalloc((void**)&OutputPathsTime,sizeof(int)*MaxPathLen));
	gpuErrchk(cudaMemset(OutputPathsTime,0,sizeof(int)*MaxPathLen));
	gpuErrchk(cudaMallocManaged((void **)&device_FitnessArray, sizeof(double)*NumRowsForPathMatrix));
	gpuErrchk(cudaMemset(device_FitnessArray,0,sizeof(double)*NumRowsForPathMatrix));
	gpuErrchk(cudaMalloc((void **)&device_TimeArray, sizeof(int)*NumRowsForPathMatrix));
	gpuErrchk(cudaMemset(device_TimeArray,0,sizeof(int)*NumRowsForPathMatrix));

	gpuErrchk(cudaMalloc((void**)&SelectionPool, sizeof(int)*PopulationSize));
	host_SelectionPool=(int*)calloc(sizeof(int),PopulationSize);
	for(int i=0;i<PopulationSize;i++)
		host_SelectionPool[i]=i;
	gpuErrchk(cudaMemcpy(SelectionPool,host_SelectionPool,sizeof(int)*PopulationSize,cudaMemcpyHostToDevice));

	int ExtraRows=NumRowsForPathMatrix-PopulationSize;

	gpuErrchk(cudaMallocManaged((void **)&ReplacementPool, sizeof(int)*ExtraRows));
	host_ReplacementPool=(int*)calloc(sizeof(int),ExtraRows);
	for(int i=PopulationSize;i<NumRowsForPathMatrix;i++)
		host_ReplacementPool[i-PopulationSize]=i;
	gpuErrchk(cudaMemcpy(ReplacementPool,host_ReplacementPool,sizeof(int)*ExtraRows,cudaMemcpyHostToDevice));
}

__global__ void updateSelectionPool(int* SelectionPool, int* ReplacementPool, int SelectionPoolSize, int ReplacementPoolSize, int NumRowsForPathMatrix)
{
	int ReplacementPoolIndex=0;
	int SelectionPoolIndex=0;
	for(int i=0;i<NumRowsForPathMatrix;i++)
	{
		if(ReplacementPoolIndex<ReplacementPoolSize && i==ReplacementPool[ReplacementPoolIndex])
			ReplacementPoolIndex++;
		else
			SelectionPool[SelectionPoolIndex++]=i;
	}
}

__global__ void resetPools(int* SelectionPool,int* ReplacementPool, int SelectionPoolSize, int NumRowsForPathMatrix)
{
	int thread = threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<NumRowsForPathMatrix)
	{	
		if(thread<SelectionPoolSize)
			SelectionPool[thread]=thread;
		else
			ReplacementPool[thread-SelectionPoolSize]=thread;
	}
}

void heapify(Pair* heap,int n)
{
	int parent = (n -1)/ 2;
	for (int i = parent; i >= 0; i--){
		int k = i; 
		Pair v = heap[k]; 
		int isheap = 0; 
		while(!isheap && 2 * k <= n){
			int j = 2 * k; 
			if (j < n) 
			{ 
				if (heap[j].data > heap[j + 1].data)
					j = j + 1; 
			}
			if (v.data <= heap[j].data)
				isheap = 1; 
			else{
				heap[k] = heap[j]; 
				k = j; 
			}
		}
		heap[k]= v; 
	}
}
void deleteNode(Pair* heap,int currentSize)
{
	Pair lastElement = heap[currentSize - 1];
	heap[0] = lastElement;
	currentSize--;
	heapify(heap, currentSize);
}
void Elimination(double* Array, int ArraySize, int* KSmallestElementsIndexArray, Pair* heap, int K) 
{
	for(int i=0;i<ArraySize;i++)
	{
		Pair x;
		x.data=Array[i];
		x.index=i;
		heap[i]=x;
	}
	heapify(heap,ArraySize-1);
	int currentSize=ArraySize;
	int n=0;
	while(K--)
	{
		int i =n -1;
		int item=heap[0].index;
		while(item< KSmallestElementsIndexArray[i] && i>=0)
		{
			KSmallestElementsIndexArray[i+1] = KSmallestElementsIndexArray[i];
			i--;
		}
		KSmallestElementsIndexArray[i+1] = item;
		n++;
		deleteNode(heap,currentSize);
		currentSize--;
	}
}

void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int CrossoverSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, AirportCoordinates* &device_SourceCoord, AirportCoordinates* &device_DestCoord,  GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* & device_Paths_size, int* &SelectionPool, int* &Selected, int* device_times, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int OutputIndex, int StartTime, double speed, int* &OutputPathsTime, int* &OutputAirTime, int* &TrafficMatrixSum,double* &device_FitnessArray, int* &device_TimeArray, int NumRowsForPathMatrix,int* ReplacementPool,int* MutationPool, Pair* &heap)
{	
	getInitPopulation<<<(PopulationSize/NumThreads)+1,NumThreads>>> (device_graph,device_arrSizes,device_Paths,device_Paths_size,Start,End,PopulationSize,time(NULL),device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,OutputIndex,device_FitnessArray,device_TimeArray);
	int SelectionPoolSize=PopulationSize;
	int MutationPoolSize=NumRowsForPathMatrix;
	int ReplacementPoolSize=NumRowsForPathMatrix-PopulationSize;
	double prevFitnessForConvergence=0.0;
	int numTimesPrevFitnessOccured=0;
	for(int genNum=0;genNum<NumberOfGenerations;genNum++)
	{
		updateSelectionPool<<<1,1>>>(SelectionPool,ReplacementPool,SelectionPoolSize,ReplacementPoolSize,NumRowsForPathMatrix);
		Shuffle<<<1,1>>>(SelectionPool,SelectionPoolSize,time(NULL));
		SelectionKernel<<<(SelectionSize/NumThreads)+1,NumThreads>>>(Selected,SelectionPool,SelectionSize,device_FitnessArray);
		Shuffle<<<1,1>>>(Selected,SelectionSize,time(NULL));
		CrossoverKernel<<<(CrossoverSize/NumThreads)+1,NumThreads>>> (Selected,device_Paths,device_Paths_size,CrossoverSize,time(NULL),device_graph, device_arrSizes,device_times,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,TrafficMatrixSum,OutputIndex,device_FitnessArray,device_TimeArray,ReplacementPool);
		Shuffle<<<1,1>>>(MutationPool,MutationPoolSize,time(NULL));
		Mutation<<<(NumberOfMutations/NumThreads)+1,NumThreads>>>(ReplacementPool, NumberOfMutations, time(NULL), device_graph, device_arrSizes, device_Paths, device_Paths_size, device_SourceCoord, device_DestCoord, SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,OutputIndex,device_FitnessArray,device_TimeArray);
		Repair<<<(NumRowsForPathMatrix/NumThreads)+1,NumThreads>>>(device_Paths, device_Paths_size, NumRowsForPathMatrix, device_graph, device_arrSizes, device_SourceCoord, device_DestCoord,  SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,OutputIndex,device_FitnessArray,device_TimeArray);
		cudaDeviceSynchronize();
		Elimination(device_FitnessArray,NumRowsForPathMatrix,ReplacementPool,heap,ReplacementPoolSize);
		ConvergenceTest(device_FitnessArray,NumRowsForPathMatrix,prevFitnessForConvergence,numTimesPrevFitnessOccured);
		if(numTimesPrevFitnessOccured==ConvergenceCutoff)
			break;
	}
	getOutput<<<1,1>>>(device_Paths, device_Paths_size, NumRowsForPathMatrix, OutputPaths, OutputPathsSizes, OutputDelays, OutputIndex, OutputPathsTime, device_graph, device_arrSizes, speed, device_times, StartTime, OutputAirTime, device_SourceCoord, device_DestCoord,device_FitnessArray,device_TimeArray);
	resetPools<<<(NumRowsForPathMatrix/NumThreads)+1,NumThreads>>>(SelectionPool,ReplacementPool,SelectionPoolSize,NumRowsForPathMatrix);
}
void resetForNextPair(int* &device_Paths, int* &device_times, int* &device_Paths_size, int* &Selected, int NumRowsForPathMatrix, int SelectionSize, int* &OutputPathsTime, double* &device_FitnessArray, int* &device_TimeArray)
{
	gpuErrchk(cudaMemset(device_Paths,0,sizeof(int)*NumRowsForPathMatrix*MaxPathLen));
	gpuErrchk(cudaMemset(device_times,0,sizeof(int)*NumRowsForPathMatrix*MaxPathLen))
	gpuErrchk(cudaMemset(device_Paths_size,0,sizeof(int)*NumRowsForPathMatrix));
	gpuErrchk(cudaMemset(Selected,0,sizeof(int)*SelectionSize));
	gpuErrchk(cudaMemset(OutputPathsTime,0,sizeof(int)*MaxPathLen));
	gpuErrchk(cudaMemset(device_FitnessArray,0,sizeof(double)*NumRowsForPathMatrix));
	gpuErrchk(cudaMemset(device_TimeArray,0,sizeof(int)*NumRowsForPathMatrix));
}
void CUDA_Free(int* &SectorTimeDict, AirportCoordinates* &device_SourceCoordArr, AirportCoordinates* &device_DestCoordArr, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &device_times, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputPathsTime, int* &OutputDelays, int* &OutputAirTime, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &host_OutputAirTime,int* &TrafficMatrixSum,double* &device_FitnessArray,int* &device_TimeArray)
{
	cudaFree(SectorTimeDict);
	cudaFree(device_SourceCoordArr);
	cudaFree(device_DestCoordArr);
	cudaFree(device_graph);
	cudaFree(device_arrSizes);
	cudaFree(device_Paths);
	cudaFree(device_Paths_size);
	cudaFree(SelectionPool);
	cudaFree(Selected);
	cudaFree(device_times);
	cudaFree(OutputPaths);
	cudaFree(OutputPathsSizes);
	cudaFree(OutputPathsTime);
	cudaFree(OutputDelays);
	cudaFree(OutputAirTime);
	cudaFree(TrafficMatrixSum);
	cudaFree(device_TimeArray);
	cudaFree(device_FitnessArray);
	free(host_SelectionPool);
	free(host_OutputPaths);
	free(host_OutputPathsSizes);
	free(host_OutputDelays);
	free(host_OutputAirTime);
	cudaDeviceReset();
}
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, int start, int end, int PopulationSize, int seed, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime, double speed, int* device_times, int* TrafficMatrixSum,int Index, double* device_FitnessArray, int* device_TimeArray)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<PopulationSize)
	{	
		getPath(device_graph, device_arrSizes, device_Paths, device_Paths_size, seed, device_SourceCoord, device_DestCoord, SectorTimeDict, start, end, thread, 0, StartTime, speed, device_times, TrafficMatrixSum, Index, device_FitnessArray, device_TimeArray);
	}
}
__device__ void getPath(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, int seed, AirportCoordinates* &device_SourceCoord, AirportCoordinates* &device_DestCoord, int* SectorTimeDict, int start, int end, int thread, int skip, int StartTime, double speed, int* device_times,int* &TrafficMatrixSum,int Index,double* device_FitnessArray,int* device_TimeArray)
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
		visited[start]=1;
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
	InitPathFitness(device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index,device_FitnessArray,device_TimeArray);
}
__device__ void InitPathFitness(int* device_Paths, int* device_Paths_size, int thread, GraphNode** device_graph, int* device_arrSizes, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime, double speed, int* device_times,int* TrafficMatrixSum, int AirportIndex,double* device_FitnessArray,int* device_TimeArray)
{
	int i=0;
	int j=0;
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
	double angle = 0;
	double TrafficFactor=1;
	int delay=0;
	double StaticCost=0;
	for (i=0;i<device_Paths_size[thread]-1;i++)
	{
		CurSec=device_Paths[Loc+i];
		NextSec=device_Paths[Loc+(i+1)];
		bool ok = 0;
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
				ok=1;
				break;
			}

		}
		if(!ok)
			printf("\n\n\n\n\n\nBAD BAD BAD BAD\n\n\n\n\n");
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
	angle/=Index;
	StaticCost=((double)(180.0-angle))/path_length;
	int InnerLoc=(device_Paths[Loc]*SectorTimeDictCols);
	double meanFitness=0;
	double maxFit=0;
	double bestTime=0;
	for(delay=0;delay<MaxDelay;delay++)
	{
		TrafficFactor=0;
		double time=device_times[Loc]+StartTime+delay;
		for(j=StartTime+delay;j<time;j++)
			TrafficFactor+=SectorTimeDict[InnerLoc+j];	
		for(i=1;i<device_Paths_size[thread];i++)
		{
			InnerLoc=(device_Paths[Loc+i]*SectorTimeDictCols);	
			for(j=time;j<time+device_times[Loc+i];j++)
			{
				TrafficFactor+=SectorTimeDict[InnerLoc+j];	
			}
			time=time+device_times[Loc+i];
		}
		double Fitness = StaticCost*(((double)(MaxDelay-delay))*((double)(*TrafficMatrixSum)-TrafficFactor));
		if(Fitness>maxFit)
		{
			maxFit=Fitness;
			bestTime=delay;
		}
		meanFitness+=Fitness;
	}
	meanFitness/=MaxDelay;
	device_FitnessArray[thread]=meanFitness;
	device_TimeArray[thread]=bestTime;
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
__global__ void update_SectorTimeDict(int* SectorTimeDict, int* OutputPaths, int* OutputDelays, int* OutputPathsSize, int Index, int StartTime, int* OutputPathsTime, int* TrafficMatrixSum)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	int PathLen=OutputPathsSize[Index];
	if(thread != 0 && thread < PathLen)
	{
		int Loc=(OutputPaths[Index*MaxPathLen+thread]*SectorTimeDictCols);
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
		int Loc=(OutputPaths[Index*MaxPathLen]*SectorTimeDictCols);
		int Sum=0;
		for(int i=StartTime+OutputDelays[Index];i<OutputPathsTime[0];i++)
		{
			SectorTimeDict[Loc+i]+=1;
			Sum+=1;
		}
		atomicAdd(TrafficMatrixSum, Sum);
	}
}
__global__ void Shuffle(int* Array,int ArraySize,int seed)
{
	curandState_t state;
	curand_init(seed, 0, 0, &state);
	for (int i = 0; i < ArraySize- 1; i++) 
	{
		int j = i + abs((int)curand(&state)) / (RAND_MAX / (ArraySize - i) + 1);
		int t = Array[j];
		Array[j] = Array[i];
		Array[i] = t;
	}
}
__global__ void SelectionKernel(int* Selected, int*SelectionPool, int SelectionSize,double* device_FitnessArray)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<SelectionSize)
	{
		int ChmNum1=SelectionPool[2*thread];
		int ChmNum2=SelectionPool[(2*thread)+1];
		if(device_FitnessArray[ChmNum1]<device_FitnessArray[ChmNum2])
			Selected[thread]=ChmNum2;
		else
			Selected[thread]=ChmNum1;
	}
}
__global__ void CrossoverKernel(int* Selected, int* device_Paths, int* device_Paths_size, int CrossoverSize,int seed,GraphNode** device_graph, int* device_arrSizes,int* device_times, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict,int StartTime,double speed,int* TrafficMatrixSum,int Index,double* device_FitnessArray, int* device_TimeArray, int* ReplacementPool)
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
		int PathLoc1=Selected[FirstIndex]*MaxPathLen;
		int PathLoc2=Selected[SecondIndex]*MaxPathLen;
		int FirstLen=device_Paths_size[Selected[FirstIndex]];
		int SecondLen=device_Paths_size[Selected[SecondIndex]];
		CumulativeTime1[0]=StartTime+device_TimeArray[Selected[FirstIndex]]+device_times[PathLoc1];
		CumulativeTime2[0]=StartTime+device_TimeArray[Selected[SecondIndex]]+device_times[PathLoc2];
		for(i=1;i<FirstLen;i++)
			CumulativeTime1[i]=CumulativeTime1[i-1]+device_times[PathLoc1+i];
		for(i=1;i<SecondLen;i++)
			CumulativeTime2[i]=CumulativeTime2[i-1]+device_times[PathLoc2+i];
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
				if(device_Paths[PathLoc1+i]==device_Paths[PathLoc2+j])
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
			int CrossIndex=abs((int)curand(&state))%CommonIndecesLen;
			int ChildOneLoc=ReplacementPool[FirstIndex]*MaxPathLen;
			int ChildTwoLoc=ReplacementPool[SecondIndex]*MaxPathLen;
			l1=CommonIndecesFirst[CrossIndex];  //l1 is index for first and l2 for second
			l2=CommonIndecesSecond[CrossIndex];
			for(i=0;i<=l1;i++)
				device_Paths[ChildOneLoc+i]=device_Paths[PathLoc1+i];
			r2=l1+1;
			for(i=l2+1;i<SecondLen;i++)
				device_Paths[ChildOneLoc+r2++]=device_Paths[PathLoc2+i];
			device_Paths_size[ReplacementPool[FirstIndex]]=r2;
			for(i=0;i<=l2;i++)
				device_Paths[ChildTwoLoc+i]=device_Paths[PathLoc2+i];
			r2=l2+1;
			for(i=l1+1;i<FirstLen;i++)
				device_Paths[ChildTwoLoc+r2++]=device_Paths[PathLoc1+i];
			device_Paths_size[ReplacementPool[SecondIndex]]=r2;	InitPathFitness(device_Paths,device_Paths_size,ReplacementPool[FirstIndex],device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index,device_FitnessArray,device_TimeArray);	InitPathFitness(device_Paths,device_Paths_size,ReplacementPool[SecondIndex],device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index,device_FitnessArray,device_TimeArray);
		}
	}
}
__global__ void Mutation(int* MutationPool, int NumberOfMutations, int seed, GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, AirportCoordinates* device_SourceCoord,  AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime,double speed,int* device_times,int* TrafficMatrixSum,int Index,double* device_FitnessArray,int* device_TimeArray)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<NumberOfMutations && device_Paths_size[MutationPool[thread]]>0)
	{
		curandState_t state;
		curand_init(seed, thread, 0, &state);
		int pathID=MutationPool[thread];
		int StartIndex=abs((int)curand(&state))%(device_Paths_size[pathID]-1);
		int Loc=pathID*MaxPathLen;
		getPath(device_graph, device_arrSizes, device_Paths, device_Paths_size, seed, device_SourceCoord, device_DestCoord, SectorTimeDict,device_Paths[Loc+StartIndex], device_Paths[Loc+(device_Paths_size[pathID]-1)], pathID, StartIndex, StartTime,speed,device_times,TrafficMatrixSum,Index,device_FitnessArray,device_TimeArray);
	}
}

__global__ void Repair(int* device_Paths, int* device_Paths_size, int NumRowsForPathMatrix, GraphNode** device_graph, int* device_arrSizes, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict,int StartTime,double speed,int* device_times,int* TrafficMatrixSum,int Index,double* device_FitnessArray,int* device_TimeArray)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<NumRowsForPathMatrix)
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
			InitPathFitness(device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_SourceCoord,device_DestCoord,SectorTimeDict,StartTime,speed,device_times,TrafficMatrixSum,Index,device_FitnessArray,device_TimeArray);
		}
	}
}
__global__ void getOutput(int* device_Paths, int* device_Paths_size, int NumRowsForPathMatrix, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int index, int* OutputPathsTime, GraphNode** device_graph, int* device_arrSizes,double speed, int* device_times,int StartTime, int* OutputAirTime, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord,double* device_FitnessArray,int* device_TimeArray)
{
	double maxF=device_FitnessArray[0];
	int path_index=0;
	int time=0;
	int i=1;
	int OutLoc=index*MaxPathLen;
	double distance=0;
	double angle =0;
	for(i=1;i<NumRowsForPathMatrix;i++)
	{
		if(device_FitnessArray[i]>maxF)
		{
			maxF=device_FitnessArray[i];
			path_index=i;
		}
	}
	time=device_TimeArray[path_index];
	int Loc=path_index*MaxPathLen;
	OutputPathsSizes[index]=device_Paths_size[path_index];
	OutputDelays[index]=time;
	int CurSec=device_Paths[Loc];
	int NextSec=device_Paths[Loc+1];
	double prevPointX=(device_SourceCoord+index)->X;
	double prevPointY=(device_SourceCoord+index)->Y;	
	double curPointX=0;
	double curPointY=0;
	double AnglePointsX[MaxPathLen+1];
	double AnglePointsY[MaxPathLen+1];
	int Index=0;
	AnglePointsX[Index]=prevPointX;
	AnglePointsY[Index++]=prevPointY;
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
				AnglePointsX[Index]=curPointX;
				AnglePointsY[Index++]=curPointY;
				prevPointX=curPointX;
				prevPointY=curPointY;
				break;
			}
		}
	}
	AnglePointsX[Index]=(device_DestCoord+index)->X;
	AnglePointsY[Index++]=(device_DestCoord+index)->Y;
	for (i=0;i<Index-2;i++)
	{
		angle+=getAngle(AnglePointsX[i],AnglePointsY[i],AnglePointsX[i+1],AnglePointsY[i+1],AnglePointsX[i+2],AnglePointsY[i+2]);
	}
	angle/=Index;
	distance+=euclidianDistance(prevPointX,prevPointY,(device_DestCoord+index)->X,(device_DestCoord+index)->Y);
	distance=(distance-0.94)/1.25;
	OutputAirTime[index]=ceil(distance/speed);
	OutputPathsTime[0]=StartTime+time+device_times[Loc];
	OutputPaths[OutLoc]=device_Paths[Loc];
	for(i=1;i<device_Paths_size[path_index];i++)
	{
		OutputPaths[OutLoc+i]=device_Paths[Loc+i];
		OutputPathsTime[i]=device_times[Loc+i]+OutputPathsTime[i-1];
	}
}
