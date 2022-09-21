typedef struct GraphNode
{
	int vertexID;
	double weight;
}GraphNode;

typedef struct SimulatorTriplet
{
	int StartPoint, EndPoint, PathIndex;
}SimulatorTriplet;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void CUDA_Init(std::string &CentroidFileName, std::string &GraphFileName, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedTime, int SelectionSize, int NumSectors, int PopulationSize, int NumODPairs, int* &OutputPaths, int* &OutputPathsSizes, double* &OutputPathsFitnesses, int* &OutputTimes, int* &host_OutputPaths, int* &host_OutputPathsSizes, double* &host_OutputPathsFitnesses, int* &host_OutputTimes);
void resetForNextPair(int* &device_Paths, int* &device_Paths_size, int* &Selected, int* &SelectedTime, int PopulationSize, int SelectionSize);
__global__ void update_SectorTimeDict(int* SectorTimeDict, int* OutputPaths, int* OutputDelays, int* OutputPathsSize, int Index, int StartTime);
void CUDA_Free(int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedTime, int* &OutputPaths, int* &OutputPathsSizes, double* &OutputPathsFitnesses, int* &OutputTimes, int* &host_OutputPaths, int* &host_OutputPathsSizes, double* &host_OutputPathsFitnesses, int* &host_OutputTimes);
void getPaths(std::vector<std::pair<int,int>> &ODPairs, std::vector<std::pair<std::vector<int>,int>> &Paths, int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, std::string& GraphFileName, std::string& CentroidFileName, std::vector<int>&times);
__device__ double getAngle(int A, int B, int C,double* device_centroids_x, double* device_centroids_y);
__device__ void InitPathFitness(double* device_Fitness, int* device_Paths, int* device_Paths_size, int thread, GraphNode** device_graph, int* device_arrSizes, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int StartTime);
__device__ void getPath(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, int seed, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int start, int end, int thread, int skip, int StartTime);
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize, int seed, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict,int StartTime);
__global__ void SelectionKernel(int* Selected, int*SelectionPool, double* device_Fitness,int SelectionSize,int*SelectedTime,int PopulationSize);
__global__ void CrossoverKernel(int* Selected, int* SelectedDelay, int* device_Paths, int* device_Paths_size, double* device_Fitness, int CrossoverSize,int seed,GraphNode** device_graph, int* device_arrSizes,double* device_centroids_x,double* device_centroids_y,int* SectorTimeDict,int StartTime);
__global__ void Mutation(int* MutationPool, int NumberOfMutations, int seed, GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int StartTime);
__global__ void Shuffle(int* SelectionPool,int SelectionPoolSize,int seed);
__global__ void CrossoverShuffle(int* Selection,int* SelectedTime,int SelectionSize,int seed);
__global__ void Repair(int* device_Paths, int* device_Paths_size, int PopulationSize, double* device_Fitness, GraphNode** device_graph, int* device_arrSizes, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict,int StartTime);
__global__ void getOutput(double* device_Fitness,int* device_Paths, int* device_Paths_size, int PopulationSize, int* OutputPaths, int* OutputPathsSizes, double* OutputPathsFitnesses, int* OutputTimes, int index);
void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int CrossoverSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y,  GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* & device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &Selected, int* &SelectedDelay, int* OutputPaths, int* OutputPathsSizes, double* OutputPathsFitnesses, int* OutputDelays, int OutputIndex,int StartTime);
void tokenize(std::string &str, char delim, std::vector<std::string> &out);
void writeOutput(std::vector<std::pair<std::vector<int>,int>>&Paths, std::string OutputFileName, int NumODPairs);
void readInput(std::vector<std::pair<int,int>>& ODPairs, std::string InputFileName,std::vector<int>& times);
void readCentroids(std::string CentroidFileName, double host_centroids_x[], double host_centroids_y[]);
void readGraph(std::string GraphFileName,GraphNode* host_graph[], int* arrSizes);
void getSimulatorMatrix(std::string MatFile,std::vector<std::pair<std::vector<int>,int>>& Paths, int NumODPairs);
