#include <iostream>
#include <vector>
#include <fstream>
#include "cuda_runtime_api.h"
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
#define NumThreads 32
#define MaxPathLen 1250
#define RAND_MAX 2147483647
#define FitnessMatrixCols 1
#define PI 3.141592653589793238
const double RadConvFactorToMultiply=180/PI;
using namespace std;
#include "GeneticAlgorithm.h"
#include "HelperCFunctions.c"

int main()
{	
	/*Input File Name*/
	string InputFileName="InputFromFrontend.txt";

	/*Output File Name*/
	string OutputFileName="OutputToSimulator.txt";

	/*Supplementary Files */
	string GraphFileName="CppGraph.txt";
	string CentroidFileName="CppCentroids.txt";

	/* GA Parameters*/
	int NumSectors=1250;
	int PopulationSize=40;
	int NumberOfMutations=1;
	int NumberOfGenerations=50;


	/* Read OD Pairs */
	vector<pair<int,int>> ODPairs;
	readInput(ODPairs,InputFileName);

	/* Call CUDA Genetic Algorithm to solve the Congestion Game*/
	int NumODPairs=ODPairs.size();
	int Paths[NumODPairs][MaxPathLen];
	getPaths(ODPairs,Paths,NumSectors,PopulationSize,NumberOfMutations,NumberOfGenerations,GraphFileName,CentroidFileName);// Input,Output

	/*Output all Paths to Output File*/
	//writeOutput(Paths,OutputFileName,NumODPairs);
	cout<<'\n';
	return 0;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void CUDA_Init(string &CentroidFileName, string &GraphFileName, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, int* &device_arrSizes, GraphNode** &device_graph, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &device_Output, int* &device_Output_size, bool* &VisitedForInit, int* &SelectionPool, int* &Selected, int* &SelectedTime, int SelectionSize, int NumSectors, int PopulationSize, int NumODPairs)
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
	cudaMalloc((void**)&SectorTimeDict,sizeof(int)*NumSectors*NumSectors);
	cudaMemset(SectorTimeDict,0,sizeof(int)*NumSectors*NumSectors);


	//RESET PER OD PAIR
	cudaMalloc((void **)&VisitedForInit, sizeof(bool)*PopulationSize*MaxPathLen);
	cudaMemset(VisitedForInit,0,sizeof(bool)*PopulationSize*MaxPathLen);
	
	cudaMalloc((void **)&device_Fitness, sizeof(double)*PopulationSize*FitnessMatrixCols); 
	cudaMemset(device_Fitness,0,sizeof(int)*sizeof(double)*PopulationSize*FitnessMatrixCols);
	
	cudaMalloc((void **)&device_Paths, sizeof(int)*PopulationSize*MaxPathLen);
	cudaMemset(device_Paths,-1,sizeof(int)*PopulationSize*MaxPathLen);
	
	cudaMalloc((void **)&device_Paths_size, sizeof(int)* PopulationSize);
	cudaMemset(device_Paths_size,0,sizeof(int)* PopulationSize);

	cudaMalloc((void**)&SelectionPool, sizeof(int)*PopulationSize);
	cudaMemset(SelectionPool,0,sizeof(int)*PopulationSize);
	
	cudaMalloc((void**)&Selected, sizeof(int)*SelectionSize);
	cudaMemset(Selected,0,sizeof(int)*SelectionSize);
	
	cudaMalloc((void**)&SelectedTime, sizeof(int)*SelectionSize);
	cudaMemset(SelectedTime,0,sizeof(int)*SelectionSize);
	
	cudaMalloc((void**)&device_Output,sizeof(int)*NumODPairs*NumSectors);
	cudaMemset(device_Output,-1,sizeof(int)*NumODPairs*NumSectors);
	
	cudaMalloc((void **)&device_Output_size, sizeof(int)*NumODPairs);
	cudaMemset(device_Output_size,0,sizeof(int)*NumODPairs);
}

void resetForNextPair(bool* &VisitedForInit, double* &device_Fitness, int* &device_Paths, int* &device_Paths_size, int* &SelectionPool, int* &Selected, int* &SelectedTime, int* &device_Output, int* &device_Output_size,  int PopulationSize)
{
	cudaMemset(VisitedForInit,0,sizeof(bool)*PopulationSize*MaxPathLen);
	cudaMemset(device_Fitness,0,sizeof(int)*sizeof(double)*PopulationSize*FitnessMatrixCols);
	cudaMemset(device_Paths,-1,sizeof(int)*PopulationSize*MaxPathLen);
	cudaMemset(device_Paths_size,0,sizeof(int)* PopulationSize);
	cudaMemset(SelectionPool,0,sizeof(int)*PopulationSize);
}

__global__ void update_SectorTimeDict(int* SectorTimeDict, int* device_Output, int* device_Output_size)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread < *device_Output_size)
	{
		int sector = device_Output[thread];
		SectorTimeDict[sector*MaxPathLen+thread]+=1;
	}
}

void getPaths(vector<pair<int,int>> &ODPairs, int Paths[][MaxPathLen], int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, string& GraphFileName, string& CentroidFileName)
{	
	int* SectorTimeDict; //2D
	double* device_centroids_x;
	double* device_centroids_y;
	int *device_arrSizes;
	GraphNode** device_graph;
	int* device_Paths; //2D
	int* device_Paths_size;
	double* device_Fitness; //2D
	int* device_Output; //2D
	int* device_Output_size;
	
	bool* VisitedForInit;
	int* SelectionPool;
	int* Selected;
	int* SelectedTime;
	int SelectionSize=PopulationSize/2;
	if(SelectionSize&1==1)
		SelectionSize+=1;
	
	double InitPopTime=0;
	double PrelimTime=0;
	double SelectionTime=0;
	double CrossoverTime=0;
	
	int NumODPairs=ODPairs.size();
	CUDA_Init(CentroidFileName, GraphFileName, SectorTimeDict, device_centroids_x, device_centroids_y, device_arrSizes, device_graph, device_Paths, device_Paths_size, device_Fitness, device_Output, device_Output_size, VisitedForInit, SelectionPool, Selected, SelectedTime, SelectionSize, NumSectors, PopulationSize, NumODPairs);
	
	for(int i=0;i<NumODPairs;i++)
	{	
		GeneticAlgorithm(NumSectors, PopulationSize, SelectionSize, NumberOfMutations, NumberOfGenerations, ODPairs[i].first, ODPairs[i].second, SectorTimeDict, device_centroids_x, device_centroids_y, device_arrSizes, device_graph, device_Paths, device_Paths_size, device_Fitness, device_Output, device_Output_size, VisitedForInit, SelectionPool, Selected, SelectedTime,  InitPopTime, PrelimTime, SelectionTime, CrossoverTime);
		
		resetForNextPair(VisitedForInit, device_Fitness, device_Paths, device_Paths_size, SelectionPool, Selected, SelectedTime, device_Output, device_Output_size,  PopulationSize);		
		//update_SectorTimeDict<<<1,NumThreads>>>(SectorTimeDict, output_path_ptr, output_path_size_ptr);
		cudaDeviceSynchronize();
	}
	cout<<"initial population time "<<InitPopTime<<'\n';
	cout<<"prelim time "<<PrelimTime<<'\n';
	cout<<"selection time "<<SelectionTime<<'\n';
	cout<<"crossver time "<<CrossoverTime<<'\n';
//	for(int i=0;i<NumODPairs;i++)
//		cudaMemcpy(Paths[i],device_Paths+i,MaxPathLen*sizeof(int),cudaMemcpyDeviceToHost);
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

__device__ void InitPathFitness(double* device_Fitness, int* device_Paths, int* device_Paths_size, int thread,GraphNode** device_graph,int* device_arrSizes, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict)
{
	double angle = 1;
	double path_length=0;
	for (int i=1;i<device_Paths_size[thread];i++)
	{
		int cur=device_Paths[thread*MaxPathLen+(i-1)];
		int to = device_Paths[thread*MaxPathLen+(i)];
		bool found=false;
		for(int j=0;j<device_arrSizes[cur];j++)
		{
			if(to==device_graph[cur][j].vertexID)
			{
				path_length+=device_graph[cur][j].weight;
				found=true;
				break;
			}
		}
		if(!found)
		{
			printf("INVALID PATH\t%d\t%d\n",cur,to);
			return;
		}
	}
	for (int i=0;i<device_Paths_size[thread]-2;i++)
		angle+=getAngle(device_Paths[thread*MaxPathLen+i],device_Paths[thread*MaxPathLen+(i+1)],device_Paths[thread*MaxPathLen+(i+2)],device_centroids_x,device_centroids_y);
	double StaticFitness=(1/path_length)*(1/angle);
	double TrafficFactor=1;
	for(int i=0;i<FitnessMatrixCols;i++)
	{
		TrafficFactor=1;
		for(int j=0;j<device_Paths_size[thread];j++)
		{
			TrafficFactor+=SectorTimeDict[device_Paths[thread*MaxPathLen+j]*MaxPathLen+(j+i)];
		}
		device_Fitness[thread*FitnessMatrixCols+i]=StaticFitness*(1/TrafficFactor)*(1/(double)(device_Paths_size[thread]-2+i));
	}
}

__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize,int seed,double* device_centroids_x, double* device_centroids_y,int* SectorTimeDict)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<PopulationSize)
	{	
		curandState_t state;
		curand_init(seed, thread, 0, &state);

		int ptr_pos=0;
		device_Paths[thread*MaxPathLen+ptr_pos++]=start;
		bool InitPath=false;
		bool visited[1250];
		int validIndex[20];
		int validIndexSize=0;
		int num_neighbors;
		int cur;
		while(!InitPath)
		{
			memset(visited,0,1250);
			visited[start]=true;
			ptr_pos=0;
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
		InitPathFitness(device_Fitness,device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict);
	}
}
__global__ void SelectionKernel(int* Selected, int*SelectionPool, double* device_Fitness,int SelectionSize,int*SelectedTime,int PopulationSize)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<SelectionSize)
	{
		double max1=0.0,max2=0.0;
		int t1=-1,t2=-1;
		for(int i=0;i<FitnessMatrixCols;i++)
		{
			if(device_Fitness[SelectionPool[2*thread]*FitnessMatrixCols+i]>max1)
			{
				max1=device_Fitness[SelectionPool[2*thread]*FitnessMatrixCols+i];
				t1=i;
			}
		}
		for(int i=0;i<FitnessMatrixCols;i++)
		{
			if(device_Fitness[SelectionPool[2*thread+1]*FitnessMatrixCols+i]>max2)
			{
				max2=device_Fitness[SelectionPool[2*thread+1]*FitnessMatrixCols+i];
				t2=i;
			}
		}
		if(max1<max2)
		{
			Selected[thread]=SelectionPool[2*thread+1];
			SelectedTime[thread]=t2;
		}
		else
		{
			Selected[thread]=SelectionPool[2*thread];
			SelectedTime[thread]=t1;
		}
	}
}

__global__ void CrossoverKernel(int* Selected, int* SelectedTime, int* device_Paths, int* device_Paths_size, double* device_Fitness, int CrossoverSize,int seed,GraphNode** device_graph, int* device_arrSizes,double* device_centroids_x,double* device_centroids_y,int* SectorTimeDict)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<CrossoverSize)
	{
		curandState_t state;
		curand_init(seed, thread, 0, &state);
		int t1=SelectedTime[2*thread];
		int t2=SelectedTime[2*thread+1];
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
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,Base,device_graph,device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict);
			InitPathFitness(device_Fitness,device_Paths,device_Paths_size,Other,device_graph,device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict);
		}
		
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
__global__ void CrossoverShuffle(int* Selection,int* SelectedTime,int SelectionSize,int seed)
{
	curandState_t state;
	curand_init(seed, 0, 0, &state);
	for (int i = 0; i < SelectionSize - 1; i++) 
	{
		int j = i + abs((int)curand(&state)) / (RAND_MAX / (SelectionSize - i) + 1);
		int c = Selection[j];
		int t = SelectedTime[j];
		Selection[j] = Selection[i];
		Selection[i] = c;
		SelectedTime[j]=SelectedTime[i];
		SelectedTime[i]=t;
	}
}


void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, int* &device_arrSizes, GraphNode** &device_graph, int* &device_Paths, int* & device_Paths_size, double* &device_Fitness, int* &device_Output, int* &device_Output_size, bool* &VisitedForInit, int* &SelectionPool, int* &Selected, int* &SelectedTime, double &InitPopTime, double &PrelimTime, double& SelectionTime, double& CrossoverTime)
{	
	getInitPopulation<<<(PopulationSize/NumThreads)+1,NumThreads>>> (device_graph,device_arrSizes,device_Paths,device_Paths_size,device_Fitness,Start,End,PopulationSize,time(NULL),device_centroids_x,device_centroids_y,SectorTimeDict);
	int genNum=0;
	int SelectionPoolSize=PopulationSize;
	int CrossoverSize=SelectionSize/2;
	int host_SelectionPool[SelectionPoolSize];
	for(int i=0;i<SelectionPoolSize;i++)
		host_SelectionPool[i]=i;
	cudaMemcpy(SelectionPool,host_SelectionPool,SelectionPoolSize*sizeof(int),cudaMemcpyHostToDevice);
	while(genNum<=NumberOfGenerations)
	{
		genNum+=1;
		Shuffle<<<1,1>>>(SelectionPool,SelectionPoolSize,time(NULL));
		SelectionKernel<<<(SelectionSize/NumThreads)+1,NumThreads>>>(Selected,SelectionPool,device_Fitness,SelectionSize,SelectedTime,PopulationSize);	
		CrossoverShuffle<<<1,1>>>(Selected,SelectedTime,SelectionSize,time(NULL));
		CrossoverKernel<<<(CrossoverSize/NumThreads)+1,NumThreads>>> (Selected,SelectedTime,device_Paths,device_Paths_size,device_Fitness,CrossoverSize,time(NULL),device_graph, device_arrSizes,device_centroids_x,device_centroids_y,SectorTimeDict);
	}
}
