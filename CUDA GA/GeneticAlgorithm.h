typedef struct GraphNode
{
	int vertexID;
	double weight;
}GraphNode;
void tokenize(string &str, char delim, vector<string> &out);
void GeneticAlgorithm(string CentroidFileName,string GraphFileName,string OutputFileName,int NumSectors,int PopulationSize, int SelectionSize, int NumberOfMutations, int NumberOfGenerations, int Start, int End,bool init,double*time_taken);
void readGraph(string GraphFileName,GraphNode* host_graph[], int* arrSizes);
void readCentroids(string CentroidFileName, double host_centroids_x[], double host_centroids_y[]);
__global__ void getInitPopulation(GraphNode** device_graph, int* device_arrSizes, int* device_Paths, int* device_Paths_size, double*device_Fitness, int start, int end, int PopulationSize,int seed,double* device_centroids_x, double* device_centroids_y);
void readInput(vector<pair<int,int>>& ODPairs, string InputFileName);

