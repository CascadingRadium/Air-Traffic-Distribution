void tokenize(std::string &str, char delim, std::vector<std::string> &out)
{
	size_t start;
	size_t end = 0;
	while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
	{
		end = str.find(delim, start);
		std::string s=str.substr(start, end - start);
		out.push_back(s);
	}
}
void writeOutput(std::vector<std::pair<std::vector<int>,int>>&Paths, std::string OutputFileName, int NumODPairs)
{
	std::ofstream file(OutputFileName);
	std::string line="";
	for(int i=0;i<NumODPairs;i++)
	{
		line="";
		for(int j=0;j<Paths[i].first.size();j++)
		{
			line+=std::to_string(Paths[i].first[j])+",";
		}
		if(line.length()>0)
			line.pop_back();
		line.push_back(' ');
		int st=Paths[i].second;
		int en=st+Paths[i].first.size();
		line+=std::to_string(st);
		line.push_back(' ');
		line+=std::to_string(en);
		if(i!=NumODPairs-1)
			line.push_back('\n');
		file<<line;
	}
	file.close();
}

void readInput(std::vector<std::pair<int,int>>& ODPairs, std::string InputFileName)
{
	std::fstream file(InputFileName);
	std::string line="";
	std::vector<std::string> tokens;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		ODPairs.push_back({stoi(tokens[0]),stoi(tokens[1])});
	}
	file.close();
}

void readCentroids(std::string CentroidFileName, double host_centroids_x[], double host_centroids_y[])
{
	std::string line="";
	std::fstream file(CentroidFileName);
	int sectorNum=0;
	std::vector<std::string> tokens;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		host_centroids_x[sectorNum]=stod(tokens[0]);
		host_centroids_y[sectorNum++]=stod(tokens[1]);
	}
	file.close();
}

void readGraph(std::string GraphFileName,GraphNode* host_graph[], int* arrSizes)
{
	std::string line="";
	std::fstream file(GraphFileName);
	std::vector<std::string> tokens;
	std::vector<std::string> pairString;
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
void getSimulatorMatrix(std::string MatFile,std::vector<std::pair<std::vector<int>,int>>& Paths, int NumODPairs)
{
	std::ofstream file(MatFile);
	int maxTime=0;
	for(auto pair:Paths)
		maxTime=max(maxTime,(int)pair.first.size()+pair.second);
	std::vector<std::vector<SimulatorTriplet>> OutputVector(maxTime,std::vector<SimulatorTriplet>());
	for(int pathIdx=0;pathIdx<NumODPairs;pathIdx++)
	{
		int startTime=Paths[pathIdx].second;
		int endTime=startTime+Paths[pathIdx].first.size();
		OutputVector[startTime].push_back({Paths[pathIdx].first[0],Paths[pathIdx].first[0],pathIdx});
		for(int time=startTime+1;time<endTime;time++)
			OutputVector[time].push_back({Paths[pathIdx].first[time-1-startTime],Paths[pathIdx].first[time-startTime],pathIdx});	
	}
	file<<std::to_string(NumODPairs)<<'\n';
	for(int i=0;i<maxTime;i++)
	{
		std::string line="";
		for(auto i:OutputVector[i])
			line+=std::to_string(i.StartPoint)+","+std::to_string(i.EndPoint)+","+std::to_string(i.PathIndex)+" ";
		if(line.length()!=0)
			line.pop_back();
		line+="\n";
		file<<line;
	}
	file.close();
}
