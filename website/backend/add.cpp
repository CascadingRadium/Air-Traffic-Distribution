#include<bits/stdc++.h>

using namespace std;

int main(int argc, char** argv)
{
    freopen("output.txt","w",stdout);
    // Json::Value rootJsonValue;
    // rootJsonValue["foo"] = "bar";

    // Json::StreamWriterBuilder builder;
    // builder["commentStyle"] = "None";
    // builder["indentation"] = "   ";

    // unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    // std::ofstream outputFileStream("output.txt");
    // writer -> write(rootJsonValue, &outputFileStream);
    //Json::Value val;
    int a=atoi(argv[1]);
    int b=atoi(argv[2]);
    cout<<a + b<<"\n";
}