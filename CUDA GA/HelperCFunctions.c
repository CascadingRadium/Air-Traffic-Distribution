/*typedef struct GraphNode*/
/*{*/
/*	int vertexID;*/
/*	double weight;*/
/*}GraphNode;*/
void tokenize(string &str, char delim, vector<string> &out)
{
	size_t start;
	size_t end = 0;
	while ((start = str.find_first_not_of(delim, end)) != string::npos)
	{
		end = str.find(delim, start);
		string s=str.substr(start, end - start);
		out.push_back(s);
	}
}
void writeOutput(vector<vector<int>>& Paths,vector<pair<int,int>>& Times, string OutputFileName, int NumODPairs)
{
	ofstream file(OutputFileName);
	string line="";
	for(int i=0;i<NumODPairs;i++)
	{
		line="";
		for(int j=0;j<Paths[i].size();j++)
		{
			line+=to_string(Paths[i][j])+",";
		}
		if(line.length()>0)
			line.pop_back();
		line.push_back(' ');
		line+=to_string(Times[i].first);
		line.push_back(' ');
		line+=to_string(Times[i].second);
		line.push_back('\n');
		file<<line;
	}
	file.close();
}

void readInput(vector<pair<int,int>>& ODPairs, string InputFileName)
{
	fstream file(InputFileName);
	string line="";
	vector<string> tokens;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		ODPairs.push_back({stoi(tokens[0]),stoi(tokens[1])});
	}
	file.close();
}

void readCentroids(string CentroidFileName, double host_centroids_x[], double host_centroids_y[])
{
	string line="";
	fstream file(CentroidFileName);
	int sectorNum=0;
	vector<string> tokens;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		host_centroids_x[sectorNum]=stod(tokens[0]);
		host_centroids_y[sectorNum++]=stod(tokens[1]);
	}
	file.close();
}

void readGraph(string GraphFileName,GraphNode* host_graph[], int* arrSizes)
{
	string line="";
	fstream file(GraphFileName);
	vector<string> tokens;
	vector<string> pairString;
	int VNum=0;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,' ',tokens);
		int StartSec=stoi(tokens[0]);
		GraphNode* Neighbors = (GraphNode*)malloc(sizeof(GraphNode)*tokens.size()-1); 
		for(int i=1;i<tokens.size();i++)
		{
			pairString.clear();
			tokenize(tokens[i],',',pairString);
			GraphNode* node = new GraphNode;
			node->vertexID=stoi(pairString[0]);
			node->weight=stod(pairString[1]);
			Neighbors[i-1]=*node;
		}
		host_graph[VNum]=Neighbors;
		arrSizes[VNum]=tokens.size()-1;
		VNum++;
	}
	file.close();	
}


void Shuffle(int *array, size_t n)
{
	if (n > 1) 
	{
		size_t i;
		for (i = 0; i < n - 1; i++) 
		{
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}

