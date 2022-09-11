#include<bits/stdc++.h>
using namespace std;

#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
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
int main()
{
	srand(time(NULL));
	int paths[]={0,1,2,3,4,5,6,7,8,9};
	for(int i=0;i<10;i++)
	{
		Shuffle(paths,10);
		for(int j=0;j<10;j++)
			cout<<paths[j]<<' ';
		cout<<"\n\n\n";
	}
	return 0;
}
