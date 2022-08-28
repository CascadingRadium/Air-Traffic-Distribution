void writeOutput(int Paths[][MaxPathLen], string OutputFileName, int NumODPairs)
{
	ofstream file(OutputFileName);
	string line="";
	for(int i=0;i<NumODPairs;i++)
	{
		line="";
		for(int j=0;j<MaxPathLen;j++)
		{
			if(Paths[i][j]!=-1)
				line+=to_string(Paths[i][j])+",";
			else
				break;
		}
		if(line.length()>0)
			line.pop_back();
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

