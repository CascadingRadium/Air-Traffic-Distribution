#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
#include<string.h>
#include<GeneticAlgorithm.h>
int cmpfunc(const void *a, const void *b)
{
	return (*(int*)b -*(int*)a ); // descending order
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
	file.close();	
}
void mutation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double* device_Fitness, int start, int end, int PopulationSize,int MutationRate)
{
	double d[device_Paths_size];
	int idx = 0;
	for(int j=0;j<device_Paths_size;j++)
	{
		d[j] = device_Fitness[j];
	}
	qsort(device_Fitness, device_Paths_size, sizeof(int), cmpfunc);
	int PopulationAvail[device_Paths_size/2];
	for(int i=device_Paths_size/2;i<device_Paths_size;i++)
	{
		int val = device_Fitness[i];
		for(int j=0;j<device_Paths_size;j++)
		{
			if(d[j] == device_Fitness[j])
			{
				PopulationAvail[idx] = device_Paths[j]; 
			}
		}
	}
	int MutationChance = (1-MutationRate);
	int Mutationprob = rand()
	if(Mutationprob >= MutationChance)
	{
		int startVertex = PopulationAvail[rand()%(device_Paths-1)];
		int ptr_pos=0;
		int[] NewPaths[1250];
		NewPaths[ptr_pos++]=startVertex;
		bool InitPath=false;
		bool visited[1250];
		int validIndex[20];
		int validIndexSize=0;
		int num_neighbors;
		int cur = startVertex;
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
			{
				memset(visited,false,1250);
				visited[startVertex]=true;
				memset(NewPaths,0,1250);
				ptr_pos = 0;
				NewPaths[ptr_pos++]=startVertex;
				continue;
			}
			else
			{
				cur=device_graph[cur][validIndex[curand(&state)%validIndexSize]].vertexID;
				visited[cur]=true;
				NewPaths[ptr_pos++]=cur;
				if(cur==end)
					InitPath=true;
					int idx1 = 0;
					for(int i=startVertex;i<device_Paths_size;i++)
					{
						device_Paths[i] = NewPaths[idx1];
						idx1++;
					}
					break;
				}

			}
		}
	}
		 
}
