#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
using namespace std;
#include "HelperCFunctions.c"
#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';

int main()
{	
	srand(time(NULL));
	/*Input File Name*/
	string InputFileName="InputFromFrontend.txt";
	/*Output File Name*/
	string OutputFileName="OutputToSimulator.txt";
	/*Supplementary Files */
	string GraphFileName="CppGraph.txt";
	string CentroidFileName="CppCentroids.txt";
	/* GA Parameters*/
	int NumSectors=1250;
	int PopulationSize=12;
	int NumberOfMutations=1;
	int NumberOfGenerations=50;
	/* Read OD Pairs */
	vector<pair<int,int>> ODPairs;
	readInput(ODPairs,InputFileName);
	/* Call CUDA Genetic Algorithm to solve the Congestion Game*/
	int NumODPairs=ODPairs.size();
	
	usleep(10000000);
	
	vector<vector<int>> Paths(NumODPairs,vector<int>());
	vector<pair<int,int>> Times(NumODPairs);
	for(int i=0;i< NumODPairs;i++)
	{
		int Pathsize=rand()%10+5;
		for(int j=0;j<Pathsize;j++)
			Paths[i].push_back(rand()%1250);
		int startTime = rand()%12;
		int endTime = startTime+Pathsize-1;
		Times[i]={startTime,endTime};
	}
	/*Output all Paths to Output File*/
	writeOutput(Paths,Times,OutputFileName,NumODPairs);
	cout<<'\n';
	return 0;
}
