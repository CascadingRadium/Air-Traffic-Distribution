#include <iostream>
#include <vector>
#include <fstream>
#include "cuda_runtime_api.h"
#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <time.h>
#define NumThreads 32
#define MaxPathLen 1250
#define RAND_MAX 2147483647
#define MaxDelay 12
#define PI 3.141592653589793238
const double RadConvFactorToMultiply=180/PI;
#include "GeneticAlgorithm1.0.h"
#include "HelperCFunctions1.0.c"
int main()
{	
	/*Input File Name*/
	std::string InputFileName="InputFromFrontend.txt";
	/*Frontend Output File Name*/
	std::string OutputToFrontendFileName="OutputToFrontend.txt";
	/*Simulator Output File Name*/
	std::string OutputToSimulatorFileName="OutputToSimulator.txt";
	/*Supplementary Files */
	std::string GraphFileName="CppGraph.txt";
	std::string CentroidFileName="CppCentroids.txt";
	/* GA Parameters*/
	int NumSectors=1250;
	int PopulationSize=4000;
	int NumberOfMutations=1;
	int NumberOfGenerations=500;
	/* Read OD Pairs */
	std::vector<std::pair<int,int>> ODPairs;
	std::vector<int> times;
	readInput(ODPairs,InputFileName,times);
	int NumODPairs=ODPairs.size();
	std::vector<std::pair<std::vector<int>,PathOutput>>Paths(NumODPairs);
	/* Call CUDA Genetic Algorithm to solve the Congestion Game*/
	getPaths(ODPairs,Paths,NumSectors,PopulationSize,NumberOfMutations,NumberOfGenerations,GraphFileName,CentroidFileName,times);// Input,Output
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess) 
		printf("CUDA error: %s\n",cudaGetErrorString(err)); 
	cudaProfilerStop();
	/*Output all Paths to Output File for the Frontend to read*/
	writeOutput(Paths,OutputToFrontendFileName,NumODPairs);
	/*Output all Paths to Output File for the Simulator to read*/
	getSimulatorMatrix(OutputToSimulatorFileName,Paths,NumODPairs);
	return 0;
}
void getPaths(std::vector<std::pair<int,int>> &ODPairs, std::vector<std::pair<std::vector<int>,PathOutput>> &Paths, int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, std::string& GraphFileName, std::string& CentroidFileName, std::vector<int>&times)
{	
	int* SectorTimeDict; //2D
	double* device_centroids_x;
	double* device_centroids_y;
	GraphNode** device_graph;
	int *device_arrSizes;
	int* device_Paths; //2D
	int* device_Paths_size;
	double* device_Fitness; //2D
	int* SelectionPool;
	int* host_SelectionPool;
	int* Selected;
	int* SelectedDelay;
	//OUTPUT RELATED
	int* OutputPaths; //2D
	int* OutputPathsSizes;
	double* OutputPathsFitnesses;
	int* OutputDelays;
	int* host_OutputPaths; //2D
	int* host_OutputPathsSizes;
	double* host_OutputPathsFitnesses;
	int* host_OutputDelays;
	int SelectionSize=PopulationSize/2;
	if(SelectionSize&1==1)
		SelectionSize+=1;
	int CrossoverSize=SelectionSize/2;	
	int NumODPairs=ODPairs.size();
	CUDA_Init(CentroidFileName, GraphFileName, SectorTimeDict, device_centroids_x, device_centroids_y, device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, SelectionPool, host_SelectionPool, Selected, SelectedDelay, SelectionSize, NumSectors, PopulationSize, NumODPairs, OutputPaths, OutputPathsSizes, OutputPathsFitnesses, OutputDelays, host_OutputPaths, host_OutputPathsSizes, host_OutputPathsFitnesses, host_OutputDelays);
	for(int i=0;i<NumODPairs;i++)
	{	
		GeneticAlgorithm(NumSectors, PopulationSize, SelectionSize, CrossoverSize, NumberOfMutations, NumberOfGenerations, ODPairs[i].first, ODPairs[i].second, SectorTimeDict, device_centroids_x, device_centroids_y, device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, SelectionPool, Selected, SelectedDelay, OutputPaths, OutputPathsSizes, OutputPathsFitnesses, OutputDelays, i, times[i]);
		update_SectorTimeDict<<<(MaxPathLen/NumThreads)+1,NumThreads>>>(SectorTimeDict, OutputPaths, OutputDelays, OutputPathsSizes, i, times[i]);
		resetForNextPair(device_Paths, device_Paths_size, Selected, SelectedDelay, PopulationSize, SelectionSize);		
	}
	cudaMemcpy(host_OutputPaths,OutputPaths,sizeof(int)*MaxPathLen*NumODPairs,cudaMemcpyDeviceToHost);
	cudaMemcpy(host_OutputPathsSizes,OutputPathsSizes,sizeof(int)*NumODPairs,cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_OutputPathsFitnesses,OutputPathsFitnesses,sizeof(double)*NumODPairs,cudaMemcpyDeviceToHost);
	cudaMemcpy(host_OutputDelays,OutputDelays,sizeof(int)*NumODPairs,cudaMemcpyDeviceToHost);
	//double cost=0;
	for(int i=0;i<NumODPairs;i++)
	{
		for(int j=0;j<host_OutputPathsSizes[i];j++)
		{
			Paths[i].first.push_back(host_OutputPaths[i*MaxPathLen+j]);
		}
		Paths[i].second.EstimatedDeparture=times[i];
		Paths[i].second.GroundHolding=host_OutputDelays[i];
		Paths[i].second.ActualDeparture=times[i]+host_OutputDelays[i];
		Paths[i].second.AerialDelay=host_OutputPathsSizes[i];
		Paths[i].second.ArrivalTime=times[i]+host_OutputDelays[i]+host_OutputPathsSizes[i];
	}
	CUDA_Free(SectorTimeDict, device_centroids_x, device_centroids_y, device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, SelectionPool, host_SelectionPool, Selected, SelectedDelay, OutputPaths, OutputPathsSizes, OutputPathsFitnesses, OutputDelays, host_OutputPaths, host_OutputPathsSizes, host_OutputPathsFitnesses, host_OutputDelays);
}
void CUDA_Init(std::string &CentroidFileName, std::string &GraphFileName, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedDelay, int SelectionSize, int NumSectors, int PopulationSize, int NumODPairs, int* &OutputPaths, int* &OutputPathsSizes, double* &OutputPathsFitnesses, int* &OutputDelays, int* &host_OutputPaths, int* &host_OutputPathsSizes, double* &host_OutputPathsFitnesses, int* &host_OutputDelays)
{
	//ONE TIME
	srand(time(NULL));
	GraphNode* host_graph[NumSectors];
	int host_arrSizes[NumSectors];
	double host_centroids_x[NumSectors];
	double host_centroids_y[NumSectors];
	readGraph(GraphFileName,host_graph,host_arrSizes);
	readCentroids(CentroidFileName,host_centroids_x,host_centroids_y);
	cudaMalloc((void **)&device_centroids_x, sizeof(double)*NumSectors);
	cudaMalloc((void **)&device_centroids_y, sizeof(double)*NumSectors);
	cudaMemcpy(device_centroids_x, host_centroids_x, sizeof(double)*(NumSectors),cudaMemcpyHostToDevice);
	cudaMemcpy(device_centroids_y, host_centroids_y, sizeof(double)*(NumSectors),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&device_arrSizes, sizeof(int)*NumSectors);	
	cudaMemcpy(device_arrSizes, host_arrSizes, sizeof(int)*(NumSectors),cudaMemcpyHostToDevice);
	cudaMalloc((void ***)&(device_graph), sizeof(GraphNode*)*NumSectors);
	for(int i=0;i<NumSectors;i++)
	{
		GraphNode* temp;
		cudaMalloc((void**)&temp, sizeof(GraphNode) * host_arrSizes[i]);
		cudaMemcpy(temp, host_graph[i], sizeof(GraphNode) * host_arrSizes[i], cudaMemcpyHostToDevice);
		cudaMemcpy(device_graph+i, &temp, sizeof(GraphNode*), cudaMemcpyHostToDevice);
	}
	cudaMalloc((void**)&SelectionPool, sizeof(int)*PopulationSize);
	host_SelectionPool=(int*)calloc(sizeof(int),PopulationSize);
	for(int i=0;i<PopulationSize;i++)
		host_SelectionPool[i]=i;
	cudaMemcpy(SelectionPool,host_SelectionPool,sizeof(int)*PopulationSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&SectorTimeDict,sizeof(int)*NumSectors*NumSectors);
	cudaMemset(SectorTimeDict,0,sizeof(int)*NumSectors*NumSectors);
	cudaMalloc((void**)&OutputPaths,sizeof(int)*MaxPathLen*NumODPairs);
	cudaMemset(OutputPaths,-1,sizeof(int)*MaxPathLen*NumODPairs);
	cudaMalloc((void**)&OutputPathsSizes,sizeof(int)*NumODPairs);
	cudaMemset(OutputPathsSizes,0,sizeof(int)*NumODPairs);
	cudaMalloc((void**)&OutputPathsFitnesses,sizeof(double)*NumODPairs); //DO NOT MEMSET -> DOUBLE
	cudaMalloc((void**)&OutputDelays,sizeof(int)*NumODPairs);
	cudaMemset(OutputDelays,-1,sizeof(int)*NumODPairs);
	host_OutputPaths=(int*)calloc(sizeof(int),NumODPairs*MaxPathLen);
	host_OutputPathsSizes=(int*)calloc(sizeof(int),NumODPairs);
	host_OutputPathsFitnesses=(double*)calloc(sizeof(double),NumODPairs);
	host_OutputDelays=(int*)calloc(sizeof(int),NumODPairs);
	//RESET PER OD PAIR
	cudaMalloc((void **)&device_Fitness, sizeof(double)*PopulationSize*(MaxDelay)); //DO NOT MEMSET -> DOUBLE
	cudaMalloc((void **)&device_Paths, sizeof(int)*PopulationSize*MaxPathLen);
	cudaMemset(device_Paths,-1,sizeof(int)*PopulationSize*MaxPathLen);
	cudaMalloc((void **)&device_Paths_size, sizeof(int)* PopulationSize);
	cudaMemset(device_Paths_size,0,sizeof(int)* PopulationSize);
	cudaMalloc((void**)&Selected, sizeof(int)*SelectionSize);
	cudaMemset(Selected,0,sizeof(int)*SelectionSize);
	cudaMalloc((void**)&SelectedDelay, sizeof(int)*SelectionSize);
	cudaMemset(SelectedDelay,0,sizeof(int)*SelectionSize);
}
void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int CrossoverSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y,  GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* & device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &Selected, int* &SelectedDelay, int* OutputPaths, int* OutputPathsSizes, double* OutputPathsFitnesses, int* OutputDelays, int OutputIndex,int StartTime)
{	
	getInitPopulation<<<(PopulationSize/NumThreads)+1,NumThreads>>> (device_graph,device_arrSizes,device_Paths,device_Paths_size,device_Fitness,Start,End,PopulationSize,time(NULL),device_centroids_x,device_centroids_y,SectorTimeDict,StartTime);
	int genNum=0;
	int SelectionPoolSize=PopulationSize;
	while(genNum<=NumberOfGenerations)
	{
		genNum+=1;
		if(NumberOfMutations==0)
			NumberOfMutations=1;
		NumberOfMutations=abs(rand())%2;
		Shuffle<<<1,1>>>(SelectionPool,SelectionPoolSize,time(NULL));
		SelectionKernel<<<(SelectionSize/NumThreads)+1,NumThreads>>>(Selected,SelectionPool,device_Fitness,SelectionSize,SelectedDelay,PopulationSize);
		CrossoverShuffle<<<1,1>>>(Selected,SelectedDelay,SelectionSize,time(NULL));
		CrossoverKernel<<<(CrossoverSize/NumThreads)+1,NumThreads>>> (Selected,SelectedDelay,device_Paths,device_Paths_size,device_Fitness,CrossoverSize,time(NULL),device_graph, device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict,StartTime);
		if(NumberOfMutations!=0)
		{
			Shuffle<<<1,1>>>(SelectionPool,SelectionPoolSize,time(NULL));
			Mutation<<<(NumberOfMutations/NumThreads)+1,NumThreads>>>(SelectionPool, NumberOfMutations, time(NULL), device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, PopulationSize, device_centroids_x, device_centroids_y, SectorTimeDict,StartTime);
		}
		Repair<<<(PopulationSize/NumThreads)+1,NumThreads>>>(device_Paths, device_Paths_size, PopulationSize, device_Fitness, device_graph, device_arrSizes, device_centroids_x, device_centroids_y,  SectorTimeDict,StartTime);
	}
	getOutput<<<1,1>>>(device_Fitness, device_Paths, device_Paths_size, PopulationSize, OutputPaths, OutputPathsSizes, OutputPathsFitnesses, OutputDelays, OutputIndex); 
}
void resetForNextPair(int* &device_Paths, int* &device_Paths_size, int* &Selected, int* &SelectedDelay, int PopulationSize, int SelectionSize)
{
	cudaMemset(device_Paths,-1,sizeof(int)*PopulationSize*MaxPathLen);
	cudaMemset(device_Paths_size,0,sizeof(int)* PopulationSize);
	cudaMemset(Selected,0,sizeof(int)*SelectionSize);
	cudaMemset(SelectedDelay,0,sizeof(int)*SelectionSize);
}
void CUDA_Free(int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedDelay, int* &OutputPaths, int* &OutputPathsSizes, double* &OutputPathsFitnesses, int* &OutputDelays, int* &host_OutputPaths, int* &host_OutputPathsSizes, double* &host_OutputPathsFitnesses, int* &host_OutputDelays)
{
	cudaFree(SectorTimeDict);
	cudaFree(device_centroids_x);
	cudaFree(device_centroids_y);
	cudaFree(device_graph);
	cudaFree(device_arrSizes);
	cudaFree(device_Paths);
	cudaFree(device_Paths_size);
	cudaFree(device_Fitness);
	cudaFree(SelectionPool);
	cudaFree(Selected);
	cudaFree(SelectedDelay);
	cudaFree(OutputPaths);
	cudaFree(OutputPathsSizes);
	cudaFree(OutputPathsFitnesses);
	cudaFree(OutputDelays);
	free(host_OutputPaths);
	free(host_OutputPathsSizes);
	free(host_OutputPathsFitnesses);
	free(host_OutputDelays);
	free(host_SelectionPool);
	cudaDeviceReset();
}
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize, int seed, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict,int StartTime)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<PopulationSize)
	{	
		getPath(device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, PopulationSize, seed, device_centroids_x, device_centroids_y, SectorTimeDict, start, end, thread, 0, StartTime);
	}
}
__device__ void getPath(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, int seed, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int start, int end, int thread, int skip, int StartTime)
{
	curandState_t state;
	curand_init(seed, thread, 0, &state);
	bool visited[MaxPathLen];
	int ptr_pos=skip;
	device_Paths[thread*MaxPathLen+ptr_pos++]=start;
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
				visited[device_Paths[thread*MaxPathLen+i]]=true;
		}
		visited[start]=true;
		ptr_pos=skip;
		device_Paths[thread*MaxPathLen+ptr_pos++]=start;
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
				device_Paths[thread*MaxPathLen+ptr_pos++]=cur;
				if(cur==end)
					InitPath=true;
			}
		}
	}
	device_Paths_size[thread]=ptr_pos;
	InitPathFitness(device_Fitness,device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict,StartTime);
}
__device__ void InitPathFitness(double* device_Fitness, int* device_Paths, int* device_Paths_size, int thread, GraphNode** device_graph, int* device_arrSizes, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int StartTime)
{
	double angle = 1;
	double path_length=0;
	for (int i=1;i<device_Paths_size[thread];i++)
	{
		int cur=device_Paths[thread*MaxPathLen+(i-1)];
		int to = device_Paths[thread*MaxPathLen+(i)];
		for(int j=0;j<device_arrSizes[cur];j++)
		{
			if(to==device_graph[cur][j].vertexID)
			{
				path_length+=device_graph[cur][j].weight;
				break;
			}
		}
	}
	for (int i=0;i<device_Paths_size[thread]-2;i++)
		angle+=getAngle(device_Paths[thread*MaxPathLen+i],device_Paths[thread*MaxPathLen+(i+1)],device_Paths[thread*MaxPathLen+(i+2)],device_centroids_x,device_centroids_y);
	double StaticFitness=(1/path_length)*(1/angle);
	double TrafficFactor=1;
	for(int delay=0;delay<MaxDelay;delay++)
	{
		TrafficFactor=1;
		for(int j=0;j<device_Paths_size[thread];j++)
		{
			TrafficFactor+=SectorTimeDict[device_Paths[thread*MaxPathLen+j]*MaxPathLen+(StartTime+j+delay)];
		}
		device_Fitness[thread*MaxDelay+delay]=StaticFitness*(1/TrafficFactor)*(1/((double)(device_Paths_size[thread]*(delay+1))));
	}
}
__device__ double getAngle(int A, int B, int C,double* device_centroids_x, double* device_centroids_y)
{
	double a = atan2(-(device_centroids_y[B]-device_centroids_y[A]),device_centroids_x[B]-device_centroids_x[A])*RadConvFactorToMultiply;
	double b = atan2(-(device_centroids_y[B]-device_centroids_y[C]),device_centroids_x[B]-device_centroids_x[C])*RadConvFactorToMultiply;
	if(abs(b-a)>180)
		return 180-(360-abs(b-a));
	else
		return 180-(abs(b-a));
}
__global__ void update_SectorTimeDict(int* SectorTimeDict, int* OutputPaths, int* OutputDelays, int* OutputPathsSize, int Index, int StartTime)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	int PathLen=OutputPathsSize[Index];
	if(thread < PathLen)
	{
		SectorTimeDict[OutputPaths[Index*MaxPathLen+thread]*MaxPathLen+(OutputDelays[Index]+StartTime+thread)]+=1;
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
		int t1=-1,t2=-1;
		for(int i=0;i<MaxDelay;i++)
		{
			if(device_Fitness[SelectionPool[2*thread]*MaxDelay+i]>max1)
			{
				max1=device_Fitness[SelectionPool[2*thread]*MaxDelay+i];
				t1=i;
			}
		}
		for(int i=0;i<MaxDelay;i++)
		{
			if(device_Fitness[SelectionPool[2*thread+1]*MaxDelay+i]>max2)
			{
				max2=device_Fitness[SelectionPool[2*thread+1]*MaxDelay+i];
				t2=i;
			}
		}
		if(max1<max2)
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
__global__ void CrossoverKernel(int* Selected, int* SelectedDelay, int* device_Paths, int* device_Paths_size, double* device_Fitness, int CrossoverSize,int seed,GraphNode** device_graph, int* device_arrSizes,double* device_centroids_x,double* device_centroids_y,int* SectorTimeDict,int StartTime)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<CrossoverSize)
	{
		curandState_t state;
		curand_init(seed, thread, 0, &state);
		int t1=SelectedDelay[2*thread];
		int t2=SelectedDelay[2*thread+1];
		int Base=Selected[2*thread];
		int Other=Selected[2*thread+1];
		int OtherStartIndex=t1-t2;
		if(t2>t1)
		{
			OtherStartIndex=t2-t1;
			Base=Selected[2*thread+1];
			Other=Selected[2*thread];
		}
		int CommonIndeces[MaxPathLen]={-1};
		int CommonIndecesLen=0;
		int OtherPos=OtherStartIndex+1;
		for(int i=1;i<device_Paths_size[Base]-1&& OtherPos<device_Paths_size[Other];i++)
		{
			if(device_Paths[Base*MaxPathLen+i]==device_Paths[Other*MaxPathLen+OtherPos])
				CommonIndeces[CommonIndecesLen++]=i;
			OtherPos++;
		}
		if(CommonIndecesLen!=0)
		{	
			int i=0;
			double avgBase=0,avgOther=0;
			for(i=0;i<MaxDelay;i++)
				avgBase+=device_Fitness[Base*MaxDelay+i];
			for(i=0;i<MaxDelay;i++)
				avgOther+=device_Fitness[Other*MaxDelay+i];
			avgBase/=MaxDelay;
			avgOther/=MaxDelay;
			int baseCopy[MaxPathLen]={-1};
			int baseFitnessCopy[MaxDelay]={-1};
			int otherCopy[MaxPathLen]={-1};
			int OtherFitnessCopy[MaxDelay]={-1};
			int BaseDeviceSizeCopy=device_Paths_size[Base];
			int OtherDeviceSizeCopy=device_Paths_size[Other];
			for(int i=0;i<MaxDelay;i++)
				baseFitnessCopy[i]=device_Fitness[Base*MaxDelay+i];
			for(int i=0;i<MaxDelay;i++)
				OtherFitnessCopy[i]=device_Fitness[Other*MaxDelay+i];
			for(i=0;i<device_Paths_size[Base];i++)
				baseCopy[i]=device_Paths[Base*MaxPathLen+i];
			for(i=0;i<device_Paths_size[Other];i++)
				otherCopy[i]=device_Paths[Other*MaxPathLen+i];
			int ChosenIndexBase=CommonIndeces[abs((int)curand(&state))%CommonIndecesLen];
			int ChosenIndexOther=ChosenIndexBase+OtherStartIndex;
			OtherPos=ChosenIndexBase+1;
			CommonIndecesLen=0;
			for(int i=OtherPos;i<device_Paths_size[Base];i++)
				CommonIndeces[CommonIndecesLen++]=device_Paths[Base*MaxPathLen+i];
			for(int i=ChosenIndexOther+1;i<device_Paths_size[Other];i++)
				device_Paths[Base*MaxPathLen+OtherPos++]=device_Paths[Other*MaxPathLen+i];
			for(int i=OtherPos;i<device_Paths[Base];i++)
				device_Paths[Base*MaxPathLen+i]=-1;
			device_Paths_size[Base]=OtherPos;
			OtherPos=ChosenIndexOther+1;
			for(int i=0;i<CommonIndecesLen;i++)
				device_Paths[Other*MaxPathLen+OtherPos++]=CommonIndeces[i];
			for(int i=OtherPos;i<device_Paths[Other];i++)
				device_Paths[Other*MaxPathLen+i]=-1;
			device_Paths_size[Other]=OtherPos;
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,Base,device_graph,device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict,StartTime);
			double newAvg=0;
			for(i=0;i<MaxDelay;i++)
				newAvg+=device_Fitness[Base*MaxDelay+i];
			newAvg/=MaxDelay;
			if(newAvg<avgBase)
			{
				for(i=0;i<BaseDeviceSizeCopy;i++)
					device_Paths[Base*MaxPathLen+i]=baseCopy[i];
				for(i=BaseDeviceSizeCopy;i<device_Paths_size[Base];i++)
					device_Paths[Base*MaxPathLen+i]=-1;
				device_Paths_size[Base]=BaseDeviceSizeCopy;
				for(i=0;i<MaxDelay;i++)
					device_Fitness[Base*MaxDelay+i]=baseFitnessCopy[i];
			}
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,Other,device_graph,device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict,StartTime);
			newAvg=0;
			for(i=0;i<MaxDelay;i++)
				newAvg+=device_Fitness[Other*MaxDelay+i];
			newAvg/=MaxDelay;
			if(newAvg<avgOther)
			{
				for(i=0;i<OtherDeviceSizeCopy;i++)
					device_Paths[Other*MaxPathLen+i]=otherCopy[i];
				for(i=OtherDeviceSizeCopy;i<device_Paths_size[Other];i++)
					device_Paths[Other*MaxPathLen+i]=-1;
				device_Paths_size[Other]=OtherDeviceSizeCopy;
				for(i=0;i<MaxDelay;i++)
					device_Fitness[Other*MaxDelay+i]=OtherFitnessCopy[i];
			}
		}
	}
}
__global__ void Mutation(int* MutationPool, int NumberOfMutations, int seed, GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int StartTime)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<NumberOfMutations)
	{
		curandState_t state;
		curand_init(seed, thread, 0, &state);
		int pathID=MutationPool[thread];
		int MutPoint=abs((int)curand(&state))%(device_Paths_size[pathID]-2);
		getPath(device_graph, device_arrSizes, device_Paths, device_Paths_size, device_Fitness, PopulationSize, seed, device_centroids_x, device_centroids_y, SectorTimeDict,device_Paths[pathID*MaxPathLen+MutPoint+1], device_Paths[pathID*MaxPathLen+(device_Paths_size[pathID]-1)], pathID, MutPoint+1, StartTime);
	}
}
__global__ void Repair(int* device_Paths, int* device_Paths_size, int PopulationSize, double* device_Fitness, GraphNode** device_graph, int* device_arrSizes, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict,int StartTime)
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
			for(int j=i;j<MaxPathLen;j++)
				device_Paths[Pos+j]=-1;
			device_Paths_size[thread]=i;
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict,StartTime);
		}
	}
}
__global__ void getOutput(double* device_Fitness,int* device_Paths, int* device_Paths_size, int PopulationSize, int* OutputPaths, int* OutputPathsSizes, double* OutputPathsFitnesses, int* OutputDelays, int index)
{
	double maxF=device_Fitness[0];
	int path_index=0;
	int time=0;
	int i=1;
	int Pos=index*MaxPathLen;
	for(i=1;i<MaxDelay*PopulationSize;i++)
	{
		if(device_Fitness[i]>maxF)
		{
			maxF=device_Fitness[i];
			path_index=i/MaxDelay;
			time=i%MaxDelay;
		}
	}
	int PathPos=path_index*MaxPathLen;
	for(i=0;i<device_Paths_size[path_index];i++)
		OutputPaths[Pos+i]=device_Paths[PathPos+i];
	OutputPathsSizes[index]=device_Paths_size[path_index];
	OutputDelays[index]=time;
	OutputPathsFitnesses[index]=maxF;
}


