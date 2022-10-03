#include <stdio.h>
#include "cuda_runtime_api.h"

int getSPcores(cudaDeviceProp devProp)
{  
	int cores = 0;
	switch (devProp.major){
		case 2: // Fermi
			if (devProp.minor == 1) cores =  48;
			else cores = 32;
			break;
		case 3: // Kepler
			cores =  192;
			break;
		case 5: // Maxwell
			cores =  128;
			break;
		case 6: // Pascal
			if ((devProp.minor == 1) || (devProp.minor == 2)) cores =  128;
			else if (devProp.minor == 0) cores =  64;
			else printf("Unknown device type\n");
			break;
		case 7: // Volta and Turing
			if ((devProp.minor == 0) || (devProp.minor == 5)) cores =  64;
			else printf("Unknown device type\n");
			break;
		case 8: // Ampere
			if (devProp.minor == 0) cores =  64;
			else if (devProp.minor == 6) cores =  128;
			else printf("Unknown device type\n");
			break;
		default:
			printf("Unknown device type\n"); 
			break;
		}
		return cores;
}

void printDevProp(cudaDeviceProp devProp)
{
	printf("Compute Capability:            %d.%d\n",  devProp.major,devProp.minor);
	printf("Name:                          %s\n",  devProp.name);
	printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
	printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
	printf("Warp size:                     %d\n",  devProp.warpSize);
	printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
	printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n",  devProp.clockRate);
	printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
	printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
	printf("Number of SPs or CUDA Cores per SM: %d\n", getSPcores(devProp));
	printf("Total Number of SPs or CUDA Cores: %d\n", getSPcores(devProp)*devProp.multiProcessorCount);
	printf("Max threads per processor: %d\n", devProp.maxThreadsPerMultiProcessor);
	return;
}
int main()
{
	int devCount;
	cudaDeviceReset();
	cudaGetDeviceCount(&devCount);
	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(devProp);
	}
	return 0;
}


