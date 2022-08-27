#include<iostream>
#include<vector>
#include<fstream>
#include "cuda_runtime_api.h"
#include <curand_kernel.h>
#include <curand.h>
using namespace std;
#include "GeneticAlgorithm.h"
#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
#define NumThreads 32
#define MaxPathLen 1250
#define PI 3.141592653589793238
const double RadConvFactorToMultiply=180/PI;
int main()
{
	cout.precision(10);
	/*Input Graph File */
	string GraphFileName="CppGraph.txt";
	string CentroidFileName="CppCentroids.txt";
	string OutputFileName="CUDA_Paths.txt";
	string InputFileName="InputFromFrontend.txt";

	/* GA Parameters*/
	int NumSectors=1250;
	int PopulationSize=4000;
	int SelectionSize=2000;
	int NumberOfMutations=1;
	int NumberOfGenerations=50;
	
	/*TEST Time*/
	double* time_taken = (double*)calloc(sizeof(double),1); 
	
	/* OD Pairs */
	vector<pair<int,int>> ODPairs;
	readInput(ODPairs,InputFileName);
	bool init=false;
	for(auto OD: ODPairs)
	{
		GeneticAlgorithm(CentroidFileName,GraphFileName,OutputFileName,NumSectors,PopulationSize,SelectionSize,NumberOfMutations,NumberOfGenerations,OD.first,OD.second,init,time_taken);
		init=true;
	}
	cout<<*time_taken;
	cout<<'\n';
	return 0;
}

void readInput(vector<pair<int,int>>& ODPairs, string InputFileName)
{
	fstream file(InputFileName);
	string line="";
	vector<string> tokens;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		ODPairs.push_back({stoi(tokens[0]),stoi(tokens[1])});
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

__device__ void PathFitness(double* device_Fitness, int* device_Paths, int* device_Paths_size, int thread,GraphNode** device_graph,int* device_arrSizes, double* device_centroids_x, double* device_centroids_y)
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
	device_Fitness[thread]=path_length;
	for (int i=0;i<device_Paths_size[thread]-2;i++)
		angle+=getAngle(device_Paths[thread*MaxPathLen+i],device_Paths[thread*MaxPathLen+(i+1)],device_Paths[thread*MaxPathLen+(i+2)],device_centroids_x,device_centroids_y);
}

__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize,int seed,double* device_centroids_x, double* device_centroids_y)
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
		PathFitness(device_Fitness,device_Paths,device_Paths_size,thread,device_graph,device_arrSizes,device_centroids_x,device_centroids_y);
	}
}

void readCentroids(string CentroidFileName, double host_centroids_x[], double host_centroids_y[])
{
	string line="";
	fstream file(CentroidFileName);
	int sectorNum=0;
	vector<string> tokens;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		host_centroids_x[sectorNum]=stod(tokens[0]);
		host_centroids_y[sectorNum++]=stod(tokens[1]);
	}
}
void GeneticAlgorithm(string CentroidFileName,string GraphFileName,string OutputFileName,int NumSectors,int PopulationSize, int SelectionSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, bool init,double* time_taken)
{	
	double* device_centroids_x;
	double* device_centroids_y;
	int *device_arrSizes;
	GraphNode** device_graph;
	int* device_Paths;
	int* device_Paths_size;
	double* device_Fitness;
	if(!init)
	{
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
	}
	cudaMallocManaged((void **)&(device_Paths), sizeof(int)*PopulationSize*MaxPathLen);
	cudaMemset(device_Paths,-1,sizeof(int)*PopulationSize*MaxPathLen);
	cudaMallocManaged((void **)&(device_Paths_size), sizeof(int)* PopulationSize);
	cudaMemset(device_Paths_size,0,sizeof(int)* PopulationSize);
	cudaMalloc((void **)&device_Fitness, sizeof(double)*PopulationSize);
	cudaMemset(device_Fitness,-1,sizeof(double)*PopulationSize);
	clock_t t;
	t = clock();
	getInitPopulation<<<(PopulationSize/NumThreads)+1,NumThreads>>> (device_graph,device_arrSizes,device_Paths,device_Paths_size,device_Fitness,Start,End,PopulationSize,time(NULL),device_centroids_x,device_centroids_y);
	cudaDeviceSynchronize();
	t = clock() - t;
	(*time_taken)+= ((double)t)/CLOCKS_PER_SEC;
}

void readGraph(string GraphFileName,GraphNode* host_graph[], int* arrSizes)
{
	string line="";
	fstream file(GraphFileName);
	vector<string> tokens;
	vector<string> pairString;
	int VNum=0;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,' ',tokens);
		int StartSec=stoi(tokens[0]);
		GraphNode* Neighbors = (GraphNode*)malloc(sizeof(GraphNode)*tokens.size()-1); 
		for(int i=1;i<tokens.size();i++)
		{
			pairString.clear();
			tokenize(tokens[i],',',pairString);
			GraphNode* node = new GraphNode;
			node->vertexID=stoi(pairString[0]);
			node->weight=stod(pairString[1]);
			Neighbors[i-1]=*node;
		}
		host_graph[VNum]=Neighbors;
		arrSizes[VNum]=tokens.size()-1;
		VNum++;
	}	

}

void tokenize(string &str, char delim, vector<string> &out)
{
	size_t start;
	size_t end = 0;
	while ((start = str.find_first_not_of(delim, end)) != string::npos)
	{
		end = str.find(delim, start);
		string s=str.substr(start, end - start);
		out.push_back(s);
	}
}
