#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime_api.h"
#include <time.h>
#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
int main(int argc, char *argv[])
{
	clock_t t;
	int a[4000];
	for(int i=0;i<4000;i++)
		a[i]=i;
	int* dev_a;
	cudaMalloc((void**)&dev_a,sizeof(int)*4000);
	t=clock();
	cudaMemcpy(dev_a,a,sizeof(int)*4000,cudaMemcpyHostToDevice);
	t=clock()-t;
	printf("%lf\n",((double)t)/CLOCKS_PER_SEC);
	printf()
	return 0;
}
