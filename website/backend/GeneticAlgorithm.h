typedef struct GraphNode
{
	int vertexID;
	double XCoord;
	double YCoord;
}GraphNode;

typedef struct SimulatorTriplet
{
	int StartPoint, EndPoint, PathIndex;
}SimulatorTriplet;
typedef struct
{
	int EstimatedDeparture;
	int GroundHolding;
	int ActualDeparture;
	int AerialDelay;
	int ArrivalTime;
	double speed;
}PathOutput;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void CUDA_Init(std::string &CentroidFileName, std::string &GraphFileName, GraphNode** host_graph, int* host_arrSizes,int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedDelay, int* &device_times ,int SelectionSize, int NumSectors, int PopulationSize, int NumODPairs, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputDelays, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &OutputPathsTime ,int* &OutputAirTime, int* &host_OutputAirTime,int* &TrafficMatrixSum);
void resetForNextPair(int* &device_Paths, int* &device_times, int* &device_Paths_size, int* &Selected, int* &SelectedDelay, int PopulationSize, int SelectionSize, int* &OutputPathsTime);
__global__ void update_SectorTimeDict(int* SectorTimeDict, int* OutputPaths, int* OutputDelays, int* OutputPathsSize, int Index, int StartTime, int* OutputPathsTime, int* TrafficMatrixSum);
void CUDA_Free(int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &SelectedDelay, int* &device_times, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputPathsTime, int* &OutputDelays, int* &OutputAirTime, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &host_OutputAirTime,int* & TrafficMatrixSum);
void getPaths(std::vector<std::pair<int,int>> &ODPairs, std::vector<std::pair<std::vector<int>,PathOutput>> &Paths, int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, std::string& GraphFileName, std::string& CentroidFileName, std::vector<int>&times, std::vector<double> &speeds);
__device__ double getAngle(double Ax,double Ay, double Bx, double By, double Cx, double Cy);
__device__ double euclidianDistance(double Point1X, double Point1Y, double Point2X, double Point2Y);
__device__ void InitPathFitness(double* device_Fitness, int* device_Paths, int* device_Paths_size, int thread, GraphNode** device_graph, int* device_arrSizes, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int StartTime, double speed, int* device_times,int* TrafficMatrixSum);
__device__ void getPath(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, int seed, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int start, int end, int thread, int skip, int StartTime, double speed, int* device_times,int* &TrafficMatrixSum);
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize, int seed, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int StartTime, double speed, int* device_times, int* TrafficMatrixSum);
__global__ void SelectionKernel(int* Selected, int*SelectionPool, double* device_Fitness,int SelectionSize,int*SelectedTime,int PopulationSize);
__global__ void CrossoverKernel(int* Selected, int* SelectedDelay, int* device_Paths, int* device_Paths_size, double* device_Fitness, int CrossoverSize,int seed,GraphNode** device_graph, int* device_arrSizes,int* device_times, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict,int StartTime,double speed,int* TrafficMatrixSum);
__global__ void Mutation(int* MutationPool, int NumberOfMutations, int seed, GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int PopulationSize, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict, int StartTime,double speed,int* device_times,int* TrafficMatrixSum);
__global__ void Shuffle(int* SelectionPool,int SelectionPoolSize,int seed);
__global__ void CrossoverShuffle(int* Selection,int* SelectedTime,int SelectionSize,int seed);
__global__ void Repair(int* device_Paths, int* device_Paths_size, int PopulationSize, double* device_Fitness, GraphNode** device_graph, int* device_arrSizes, double* device_centroids_x, double* device_centroids_y, int* SectorTimeDict,int StartTime,double speed,int* device_times,int* TrafficMatrixSum);
__global__ void getOutput(double* device_Fitness,int* device_Paths, int* device_Paths_size, int PopulationSize, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int index, int* OutputPathsTime, GraphNode** device_graph, int* device_arrSizes,double speed, int* device_times,int StartTime, int* OutputAirTime, double* device_centroids_x, double* device_centroids_y);
void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int CrossoverSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y,  GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* & device_Paths_size, double* &device_Fitness, int* &SelectionPool, int* &Selected, int* &SelectedDelay, int* device_times, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int OutputIndex, int StartTime, double speed, int* &OutputPathsTime, int* &OutputAirTime, int* &TrafficMatrixSum);
void writeOutput(std::vector<std::pair<std::vector<int>,PathOutput>> &Paths, std::string OutputFileName, int NumODPairs);
void readInput(std::vector<std::pair<int,int>>& ODPairs, std::string InputFileName, std::vector<int>& times, std::vector<int>& speeds);
void readCentroids(std::string CentroidFileName, double host_centroids_x[], double host_centroids_y[]);
void readGraph(std::string GraphFileName,GraphNode* host_graph[], int* arrSizes);
void readGA_Params(int &PopulationSize, int &NumberOfMutations, int &NumberOfGenerations, std::string &GA_ParametersFileName);
