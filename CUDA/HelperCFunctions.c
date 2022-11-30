void readGA_Params(int &PopulationSize, int &NumberOfMutations, int &NumberOfGenerations, std::string &GA_ParametersFileName)
{
	std::fstream file(GA_ParametersFileName);
	std::string line="";
	getline(file,line);
	PopulationSize=stoi(line);
	getline(file,line);
	NumberOfMutations=stoi(line);
	getline(file,line);
	NumberOfGenerations=stoi(line);
	file.close();
}

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

void readRunways(std::string RunwayFileName, std::unordered_map<std::string,std::pair<int,int>> &AirportRunways)
{
	std::fstream file(RunwayFileName);
	std::vector<std::string> tokens;
	std::string line="";
	int index=0;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		AirportRunways[tokens[0]]={stoi(tokens[1]),index};
		index++;
	}
	file.close();
}
void writeOutput(std::vector<std::pair<std::vector<int>,PathOutput>>&Paths, std::string OutputToFrontendFileName, std::string AerGDFileName, int NumODPairs, std::vector<std::vector<int>> &AirspaceTraffic, std::string AirspaceTrafficFileName, std::vector<std::vector<int>> &AirportTraffic, std::string AirportTrafficFileName, std::unordered_map<std::string,std::pair<int,int>> &AirportRunways)
{
	std::vector<std::pair<std::string,std::pair<int,int>>> AirportRunwayVector(AirportRunways.begin(),AirportRunways.end());
	sort(AirportRunwayVector.begin(),AirportRunwayVector.end(),[](auto &A, auto &B){return A.second.second<B.second.second;});
	std::ofstream OtoFfile(OutputToFrontendFileName);
	std::ofstream AerGDfile(AerGDFileName);
	std::ofstream ASTfile(AirspaceTrafficFileName);
	std::ofstream APTfile(AirportTrafficFileName);
	
	AerGDfile<<"Aerial Time,Ground Holding\n";
	std::string line="";
	std::string AerLine="";
	for(int i=0;i<NumODPairs;i++)
	{
		line="";
		AerLine="";
		for(int j=0;j<Paths[i].first.size();j++)
		{
			line+=std::to_string(Paths[i].first[j])+",";
		}
		if(line.length()>0)
			line.pop_back();
		line+=(","+std::to_string(Paths[i].second.EstimatedDeparture)+","+std::to_string(Paths[i].second.GroundHolding)+","+std::to_string(Paths[i].second.ActualDeparture)+","+std::to_string(Paths[i].second.AerialDelay)+","+std::to_string(Paths[i].second.ArrivalTime)+","+std::to_string(Paths[i].second.speed)+","+Paths[i].second.StartICAO+","+Paths[i].second.EndICAO);
		AerLine+=(std::to_string(Paths[i].second.AerialDelay)+","+std::to_string(Paths[i].second.GroundHolding)+"\n");
		AerGDfile<<AerLine;		
		if(i!=NumODPairs-1)
			line.push_back('\n');
		OtoFfile<<line;
	}
	OtoFfile.close();
	AerGDfile.close();
	line="";
	for(int i=0;i<AirspaceTraffic.size();i++)
	{
		for(int j=0;j<SectorTimeDictCols;j++)
			line+=(std::to_string(AirspaceTraffic[i][j])+",");
		line.pop_back();
		line+="\n";
	}
	ASTfile<<line;
	line="";
	for(int i=0;i<AirportTraffic.size();i++)
	{
		line+=(AirportRunwayVector[i].first+" "+std::to_string(AirportRunwayVector[i].second.first)+" ");
		for(int j=0;j<SectorTimeDictCols;j++)
			line+=(std::to_string(AirportTraffic[i][j])+",");
		line.pop_back();
		line+="\n";
	}
	APTfile<<line;
	ASTfile.close();
	APTfile.close();
}

void readInput(std::vector<std::pair<Airport,Airport>>& ODPairs, std::string InputFileName, std::vector<int>& times, std::vector<double>& speeds)
{
	std::fstream file(InputFileName);
	std::string line="";
	std::vector<std::string> tokens;
	std::vector<std::string> innertoken;
	while(getline(file,line))
	{
		tokens.clear();
		tokenize(line,',',tokens);
		innertoken.clear();
		tokenize(tokens[0],' ',innertoken);
		Airport source;
		source.sector=stoi(innertoken[0]);
		source.ICAO=innertoken[1];
		source.X=stod(innertoken[2]);
		source.Y=stod(innertoken[3]);
		innertoken.clear();
		tokenize(tokens[1],' ',innertoken);
		Airport dest;
		dest.sector=stoi(innertoken[0]);
		dest.ICAO=innertoken[1];
		dest.X=stod(innertoken[2]);
		dest.Y=stod(innertoken[3]);
		ODPairs.push_back({source,dest});
		times.push_back(stoi(tokens[2]));
		speeds.push_back(stod(tokens[3]));
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
			node->XCoord=stod(pairString[1]);
			node->YCoord=stod(pairString[2]);
			Neighbors[i-1]=*node;
		}
		host_graph[VNum]=Neighbors;
		arrSizes[VNum]=tokens.size()-1;
		VNum++;
	}
	file.close();	
}
