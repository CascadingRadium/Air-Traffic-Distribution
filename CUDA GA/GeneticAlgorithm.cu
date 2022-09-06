#include <iostream>
#include <vector>
#include <fstream>
#include "cuda_runtime_api.h"
#include <curand_kernel.h>
#include <curand.h>
#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
#define NumThreads 32
#define MaxPathLen 1250
#define FitnessMatrixCols 12
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
	int PopulationSize=49;
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
	writeOutput(Paths,OutputFileName,NumODPairs);
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
void CUDA_Init(string &CentroidFileName, string &GraphFileName, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, int* &device_arrSizes, GraphNode** &device_graph, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &device_Output, int* &device_Output_size,int NumSectors, int PopulationSize, int NumODPairs, bool* &Valid, int* &SelectionPool, int* &Selected, int* &SelectedTime,int* &PopSizeNoDupli,int SelectionSize)
{
	//ONE TIME
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
	cudaMalloc((void**)&device_Output,sizeof(int)*NumODPairs*NumSectors);
	cudaMemset(device_Output,-1,sizeof(int)*NumODPairs*NumSectors);
	cudaMalloc((void **)&device_Output_size, sizeof(int)*NumODPairs);
	cudaMemset(device_Output_size,0,sizeof(int)*NumODPairs);

	//RESET PER OD PAIR
	cudaMalloc((void **)&device_Fitness, sizeof(double)*PopulationSize*FitnessMatrixCols); 
	cudaMalloc((void **)&(device_Paths), sizeof(int)*PopulationSize*MaxPathLen);
	cudaMemset(device_Paths,-1,sizeof(int)*PopulationSize*MaxPathLen);
	cudaMalloc((void **)&(device_Paths_size), sizeof(int)* PopulationSize);
	cudaMemset(device_Paths_size,0,sizeof(int)* PopulationSize);
	cudaMallocManaged((void**)&Valid, sizeof(bool)*PopulationSize);
	cudaMemset(Valid,1,PopulationSize);
	cudaMalloc((void**)&SelectionPool, sizeof(int)*PopulationSize);
	cudaMalloc((void**)&Selected, sizeof(int)*SelectionSize);
	cudaMalloc((void**)&SelectedTime, sizeof(int)*SelectionSize);
	cudaMallocManaged((void**)&PopSizeNoDupli, sizeof(int));
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
	
	bool* Valid;
	int* SelectionPool;
	int* Selected;
	int* SelectedTime;
	int* PopSizeNoDupli;
	int SelectionSize=PopulationSize/2;
	
	int NumODPairs=ODPairs.size();
	CUDA_Init(CentroidFileName, GraphFileName, SectorTimeDict, device_centroids_x, device_centroids_y, device_arrSizes, device_graph, device_Paths, device_Paths_size, device_Fitness, device_Output, device_Output_size ,NumSectors, PopulationSize,NumODPairs,Valid,SelectionPool,Selected,SelectedTime,PopSizeNoDupli,SelectionSize);
	for(int i=0;i<NumODPairs;i++)
	{
		int* output_path_ptr=device_Output+(i*NumSectors);
		int* output_path_size_ptr=device_Output_size+i;	GeneticAlgorithm(NumSectors,PopulationSize,NumberOfMutations,NumberOfGenerations,ODPairs[i].first,ODPairs[i].second,SectorTimeDict,device_centroids_x,device_centroids_y,device_arrSizes,device_graph,device_Paths,device_Fitness,output_path_ptr,output_path_size_ptr,device_Paths_size,Valid,SelectionPool,Selected,SelectedTime,PopSizeNoDupli);
		cudaMemset(device_Paths,-1,sizeof(int)*PopulationSize*MaxPathLen);
		cudaMemset(device_Paths_size,0,sizeof(int)* PopulationSize);
		update_SectorTimeDict<<<1,NumThreads>>>(SectorTimeDict, output_path_ptr, output_path_size_ptr);
		cudaDeviceSynchronize();
	}
	for(int i=0;i<NumODPairs;i++)
		cudaMemcpy(Paths[i],device_Paths+i,MaxPathLen*sizeof(int),cudaMemcpyDeviceToHost);
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
					cur=device_graph[cur][validIndex[curand(&state)%validIndexSize]].vertexID;
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

__global__ void Prelim(int* device_Paths,bool* Valid,int PopulationSize,int Cur,int*device_Paths_size,int* PopSizeNoDupli)
{
	int thread= threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<PopulationSize && Valid[Cur] && Cur != thread && Valid[thread] && device_Paths_size[Cur]==device_Paths_size[thread])
	{
		int* BasePath=device_Paths+(Cur*MaxPathLen);
		int* ComparePath=device_Paths+(thread*MaxPathLen);
		int x=0;
		for(int i=0;i<MaxPathLen;i++)
			if(BasePath[i]==ComparePath[i])
				x++;
			else
				break;
		if(x==MaxPathLen)
		{
			Valid[thread]=false;
			atomicSub(PopSizeNoDupli,1);
		}
	}
}

__global__ void SelectionKernel(int* Selected, int*SelectionPool, double* device_Fitness,int SelectionSize,int*SelectedTime,int* PopSizeNoDupli)
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


void GeneticAlgorithm(int NumSectors,int PopulationSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, int* &device_arrSizes, GraphNode** &device_graph, int* &device_Paths, double* &device_Fitness, int* &device_Output, int* &device_Output_size, int* & device_Paths_size, bool* &Valid,int* &SelectionPool, int* &Selected,int* &SelectedTime,int* &PopSizeNoDupli)
{	
	cout.precision(10);
	getInitPopulation<<<(PopulationSize/NumThreads)+1,NumThreads>>> (device_graph,device_arrSizes,device_Paths,device_Paths_size,device_Fitness,Start,End,PopulationSize,time(NULL),device_centroids_x,device_centroids_y,SectorTimeDict);
	cudaDeviceSynchronize();
	//	for(int i=0;i<PopulationSize;i++)
	//	{
	//		for(int j=0;j<FitnessMatrixCols;j++)
	//			cout<<device_Fitness[i*FitnessMatrixCols+j]<<' ';
	//		printf("\n");
	//	}
	int genNum=0;
	cudaMemset(Valid,1,PopulationSize);
	int SelectionPoolSize=PopulationSize;
	int SelectionSize=PopulationSize/2;
	int* host_SelectionPool = (int*)malloc(sizeof(int)*SelectionPoolSize);
	*PopSizeNoDupli=PopulationSize;
	int tempforPool=0;
	while(genNum<=NumberOfGenerations)
	{
		//Prelim ->Eliminate duplicate chromosomes
		for(int i=0;i<PopulationSize;i++)
			Prelim<<<(PopulationSize/NumThreads)+1,NumThreads>>>(device_Paths,Valid,PopulationSize,i,device_Paths_size,PopSizeNoDupli);
		cudaDeviceSynchronize();
		SelectionSize=*PopSizeNoDupli/2;
		if(SelectionSize&1)
			SelectionSize+=1;
		SelectionPoolSize=*PopSizeNoDupli;
		memset(host_SelectionPool,-1,SelectionPoolSize*sizeof(int));
		cudaMemset(SelectionPool,-1,SelectionPoolSize*sizeof(int));
		cudaMemset(Selected,-1,SelectionSize*sizeof(int)); 
		cudaMemset(SelectedTime,-1,SelectionSize*sizeof(int));
		tempforPool=0;
		for(int i=0;i<SelectionPoolSize;i++)
		{
			if(Valid[i])
				host_SelectionPool[tempforPool++]=i;	
		}
		shuffle(host_SelectionPool,SelectionPoolSize);
		cudaMemcpy(SelectionPool,host_SelectionPool,SelectionPoolSize,cudaMemcpyHostToDevice);
		SelectionKernel<<<(SelectionSize/NumThreads)+1,NumThreads>>>(Selected,SelectionPool,device_Fitness,SelectionSize,SelectedTime,PopSizeNoDupli);
		cudaDeviceSynchronize();
		break;
	}
}
