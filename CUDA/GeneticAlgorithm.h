typedef struct GraphNode
{
	int vertexID;
	double XCoord;
	double YCoord;
}GraphNode;

typedef struct Pair{
	int Index;
	double Value;
}Pair;

struct PairCmp 
{
	__host__ __device__ bool operator()(const Pair& o1, const Pair& o2) 
	{
		return o1.Value < o2.Value;
	}
};

typedef struct delayType
{
    int X[MaxDelay];
    int Y[MaxDelay];
    int size;
    int p;
}delayType;

typedef struct slopeData
{
    double slopeArray[MaxDelay];
    int size;
}slopeData;

typedef struct minPair
{
	int minValue;
	int minIdx;
}minPair;

typedef struct PathOutput
{
	int EstimatedDeparture;
	int GroundHolding;
	int ActualDeparture;
	int AerialDelay;
	int ArrivalTime;
	double speed;
	std::string StartICAO;
	std::string EndICAO;
}PathOutput;

typedef struct Airport
{
	int sector;
	std::string ICAO;
	double X;
	double Y;
}Airport;

typedef struct AirportCoordinates
{
	double X;
	double Y;
	int MatrixIndex;
	int NumRunways;
}AirportCoordinates;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void CUDA_Init(std::string &GraphFileName, GraphNode** host_graph, int* host_arrSizes,int* &SectorTimeDict, AirportCoordinates* &device_SourceCoordArr, AirportCoordinates* &device_DestCoordArr, AirportCoordinates* host_SourceCoordArr, AirportCoordinates* host_DestCoordArr, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &device_times ,int SelectionSize, int NumSectors, int PopulationSize, int NumODPairs, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputDelays, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &OutputPathsTime ,int* &OutputAirTime, int* &host_OutputAirTime,int* &TrafficMatrixSum, double* &device_FitnessArray, double* &host_FitnessArray, int* &device_TimeArray,int NumRowsForPathMatrix,int* &ReplacementPool,int* &host_ReplacementPool,int* &MutationPool,int* &host_MutationPool, Pair* &d_PairArray, int* &host_SectorTimeDict, int* &AirportMatrix, int* &host_AirportMatrix, int NumAirports);
void resetForNextPair(int* &device_Paths, int* &device_times, int* &device_Paths_size, int* &Selected, int NumRowsForPathMatrix, int SelectionSize, int* &OutputPathsTime, double* &device_FitnessArray, int* &device_TimeArray);
__global__ void update_SectorTimeDict(int* SectorTimeDict, int* OutputPaths, int* OutputDelays, int* OutputPathsSize, int Index, int StartTime, int* OutputPathsTime, int* TrafficMatrixSum,int* AirportMatrix, int SrcAirMatIndex, int DstAirMatIndex);
void CUDA_Free(int* &SectorTimeDict, AirportCoordinates* &device_SourceCoordArr, AirportCoordinates* &device_DestCoordArr, GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* &device_Paths_size, int* &SelectionPool, int* &host_SelectionPool, int* &Selected, int* &device_times, int* &OutputPaths, int* &OutputPathsSizes, int* &OutputPathsTime, int* &OutputDelays, int* &OutputAirTime, int* &host_OutputPaths, int* &host_OutputPathsSizes, int* &host_OutputDelays, int* &host_OutputAirTime,int* &TrafficMatrixSum,double* &device_FitnessArray,int* &device_TimeArray,int* &ReplacementPool,int* &host_ReplacementPool,int* &MutationPool,int* &host_MutationPool,Pair* &d_PairArray, double* host_FitnessArray,int* &host_SectorTimeDict,int* &AirportMatrix,int* host_AirportMatrix);
void getPaths(std::vector<std::pair<Airport,Airport>> &ODPairs, int NumODPairs, int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, std::string& GraphFileName, std::vector<int>& times, std::vector<double> &speeds, std::unordered_map<std::string,std::pair<int,int>> &AirportRunways, int NumAirports, std::vector<std::pair<std::vector<int>,PathOutput>> &Paths, std::vector<std::vector<int>> &AirspaceTraffic,std::vector<std::vector<int>> &AirportTraffic);
__device__ double getAngle(double Ax,double Ay, double Bx, double By, double Cx, double Cy);
__device__ double euclidianDistance(double Point1X, double Point1Y, double Point2X, double Point2Y);
__device__ void InitPathFitness(int* device_Paths, int* device_Paths_size, int thread, GraphNode** device_graph, int* device_arrSizes, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime, double speed, int* device_times,int* TrafficMatrixSum, int AirportIndex,double* device_FitnessArray,int* device_TimeArray, int* AirportMatrix);
__device__ void getPath(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, int seed, AirportCoordinates* &device_SourceCoord, AirportCoordinates* &device_DestCoord, int* SectorTimeDict, int start, int end, int thread, int skip, int StartTime, double speed, int* device_times,int* &TrafficMatrixSum,int Index,double* device_FitnessArray,int* device_TimeArray,int* AirportMatrix);
__device__ int getNewDelay(int* arr,int n);
__device__ slopeData getSlope(int x1,int y1,int* X,int n,int* Y,int k);
__device__ delayType getDelay(int* data,int n);
__device__ minPair minSlope(double* slopeArray,int n);
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, int start, int end, int PopulationSize, int seed, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime, double speed, int* device_times, int* TrafficMatrixSum,int Index, double* device_FitnessArray, int* device_TimeArray, int* AirportMatrix);
__global__ void SelectionKernel(int* Selected, int*SelectionPool, int SelectionSize,double* device_FitnessArray);
__global__ void CrossoverKernel(int* Selected, int* device_Paths, int* device_Paths_size, int CrossoverSize,int seed,GraphNode** device_graph, int* device_arrSizes,int* device_times, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict,int StartTime,double speed,int* TrafficMatrixSum,int Index,double* device_FitnessArray, int* device_TimeArray, int* ReplacementPool,int* AirportMatrix);
__global__ void Mutation(int* MutationPool, int NumberOfMutations, int seed, GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, AirportCoordinates* device_SourceCoord,  AirportCoordinates* device_DestCoord, int* SectorTimeDict, int StartTime,double speed,int* device_times,int* TrafficMatrixSum,int Index,double* device_FitnessArray,int* device_TimeArray,int* AirportMatrix);
__global__ void Shuffle(int* Array,int ArraySize,int seed);
__global__ void Repair(int* device_Paths, int* device_Paths_size, int NumRowsForPathMatrix, GraphNode** device_graph, int* device_arrSizes, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord, int* SectorTimeDict,int StartTime,double speed,int* device_times,int* TrafficMatrixSum,int Index,double* device_FitnessArray,int* device_TimeArray,int* AirportMatrix);
__global__ void getOutput(int* device_Paths, int* device_Paths_size, int NumRowsForPathMatrix, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int index, int* OutputPathsTime, GraphNode** device_graph, int* device_arrSizes,double speed, int* device_times,int StartTime, int* OutputAirTime, AirportCoordinates* device_SourceCoord, AirportCoordinates* device_DestCoord,double* device_FitnessArray,int* device_TimeArray);
__global__ void updateSelectionPool(int* SelectionPool, int* ReplacementPool, int SelectionPoolSize, int ReplacementPoolSize, int NumRowsForPathMatrix);
__global__ void resetPools(int* SelectionPool,int* ReplacementPool, int SelectionPoolSize, int NumRowsForPathMatrix);
__global__ void updateReplacementPool(int* ReplacementPool, Pair* d_PairArray,int ReplacementPoolSize);
__global__ void makeArray(double* device_FitnessArray, Pair* d_PairArray, int NumRowsForPathMatrix);
void ConvergenceTest(double* host_FitnessArray,int NumRowsForPathMatrix,double &prevFitnessForConvergence, int& numTimesPrevFitnessOccured);
void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int CrossoverSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, AirportCoordinates* &device_SourceCoord, AirportCoordinates* &device_DestCoord,  GraphNode** &device_graph, int* &device_arrSizes, int* &device_Paths, int* & device_Paths_size, int* &SelectionPool, int* &Selected, int* device_times, int* OutputPaths, int* OutputPathsSizes, int* OutputDelays, int OutputIndex, int StartTime, double speed, int* &OutputPathsTime, int* &OutputAirTime, int* &TrafficMatrixSum,double* &device_FitnessArray, double* &host_FitnessArray, int* &device_TimeArray, int NumRowsForPathMatrix,int* ReplacementPool,int* MutationPool, Pair* &d_PairArray, int* &AirportMatrix);
void updateOrginDest(AirportCoordinates* &device_SourceCoord,AirportCoordinates* &device_DestCoord,Airport &Start, Airport &End, AirportCoordinates* &host_temp);
void writeOutput(std::vector<std::pair<std::vector<int>,PathOutput>>&Paths, std::string OutputToFrontendFileName, std::string AerGDFileName, int NumODPairs, std::vector<std::vector<int>> &AirspaceTraffic, std::string AirspaceTrafficFileName, std::vector<std::vector<int>> &AirportTraffic, std::string AirportTrafficFileName);
void readInput(std::vector<std::pair<AirportCoordinates,AirportCoordinates>>& ODPairs, std::string InputFileName, std::vector<int>& times, std::vector<double>& speeds);
void readCentroids(std::string CentroidFileName, double host_centroids_x[], double host_centroids_y[]);
void readGraph(std::string GraphFileName,GraphNode* host_graph[], int* arrSizes);
void readGA_Params(int &PopulationSize, int &NumberOfMutations, int &NumberOfGenerations, std::string &GA_ParametersFileName);
