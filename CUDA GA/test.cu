#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime_api.h"
#include <time.h>
using namespace std;
#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
int main(int argc, char *argv[])
{
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);
	cout<<"using "<<properties.multiProcessorCount<<" multiprocessors"<<endl;
	cout<<"max threads per processor: "<<properties.maxThreadsPerMultiProcessor<<endl;
	return 0;
}
