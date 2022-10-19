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
void writeOutput(std::vector<std::pair<std::vector<int>,PathOutput>>&Paths, std::string OutputFileName, std::vector<int> &TrafficFactorMetric, std::string TrafficFactorMetricFileName, std::string AerGDFileName, int NumODPairs)
{
	std::ofstream file(OutputFileName);
	std::ofstream TFfile(TrafficFactorMetricFileName);
	std::ofstream AerGDfile(AerGDFileName);
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
		line.push_back(',');
		line+=std::to_string(Paths[i].second.EstimatedDeparture);
		line.push_back(',');
		line+=std::to_string(Paths[i].second.GroundHolding);
		line.push_back(',');
		line+=std::to_string(Paths[i].second.ActualDeparture);
		line.push_back(',');
		line+=std::to_string(Paths[i].second.AerialDelay);
		line.push_back(',');
		line+=std::to_string(Paths[i].second.ArrivalTime);
		line.push_back(',');
		line+=std::to_string(Paths[i].second.speed);
		line.push_back(',');
		line+=Paths[i].second.StartICAO;
		line.push_back(',');
		line+=Paths[i].second.EndICAO;
		AerLine+=std::to_string(Paths[i].second.AerialDelay);
		AerLine.push_back(',');
		AerLine+=std::to_string(Paths[i].second.GroundHolding);
		AerLine.push_back('\n');
		AerGDfile<<AerLine;		
		if(i!=NumODPairs-1)
			line.push_back('\n');
		file<<line;
	}
	file.close();
	AerGDfile.close();
	line="";
	for(int i=0;i<TrafficFactorMetric.size();i++)
	{
		line+=std::to_string(TrafficFactorMetric[i]);
		line+="\n";
	}
	TFfile<<line;
	TFfile.close();
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
