typedef struct GraphNode
{
	int vertexID;
	double weight;
}GraphNode;
void tokenize(string &str, char delim, vector<string> &out);
void readGraph(string GraphFileName,GraphNode* host_graph[], int* arrSizes);
void readCentroids(string CentroidFileName, double host_centroids_x[], double host_centroids_y[]);
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize,int seed,double* device_centroids_x, double* device_centroids_y);
void readInput(vector<pair<int,int>>& ODPairs, string InputFileName);
void getPaths(vector<pair<int,int>> &ODPairs, int Paths[][MaxPathLen], int NumSectors, int PopulationSize, int NumberOfMutations, int NumberOfGenerations, string& GraphFileName, string& CentroidFileName);
void writeOutput(vector<vector<int>>& Paths,vector<pair<int,int>>& Times, string OutputFileName, int NumODPairs);
void GeneticAlgorithm(int NumSectors, int PopulationSize, int SelectionSize, int CrossoverSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End, int* &SectorTimeDict, double* &device_centroids_x, double* &device_centroids_y, int* &device_arrSizes, GraphNode** &device_graph, int* &device_Paths, int* & device_Paths_size, double* &device_Fitness, int* &host_Output, int* &host_Output_size, int* &SelectionPool, int* &Selected, int* &SelectedTime);
