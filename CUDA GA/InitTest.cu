#include<iostream>
#include<vector>
using namespace std;

void initArray(int* &Array)
{
	cudaMallocManaged((void**)&Array,sizeof(int)*8);
	for(int i=0;i<8;i++)
		Array[i]=i+1;
	
}
int main()
{
	int* Array;
	initArray(Array);
	for(int i=0;i<8;i++)
		cout<<Array[i]<<' ';
	cout<<'\n';
}
