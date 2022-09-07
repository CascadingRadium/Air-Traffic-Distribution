#include<bits/stdc++.h>
using namespace std;
#include"HelperCFunctions.c"

#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
int main()
{
	int paths[]={1,2,3,4,5,6,7,8,9};
	int times[]={1,2,3,4,5,6,7,8,9};
	CrossoverShuffle(paths,times,9);
	for(int i=0;i<9;i++)
		cout<<paths[i]<<' ';
	cout<<'\n';
	for(int i=0;i<9;i++)
		cout<<times[i]<<' ';
	cout<<'\n';
	return 0;
}
