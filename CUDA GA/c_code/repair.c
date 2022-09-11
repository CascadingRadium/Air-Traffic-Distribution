
#include<stdio.h>
#include<string.h>
// IDX is the thread id , thread is the formula being used 
void repair(int * device_Paths,int * device_Paths_size,int* dict,int idx)

{
	// int thread= threadIdx.x+(blockIdx.x*blockDim.x);

	int thread=idx*10;
	int new_path_index=thread,index=thread;

	for(int i=thread;i<thread+device_Paths_size[idx];i++)
	{


		dict[device_Paths[i]]=i;
	}	

	while(index<device_Paths_size[idx]+thread)
	{
		device_Paths[new_path_index]=device_Paths[index];
		new_path_index++;
		index=dict[device_Paths[index]]+1;

	}

	int new_len=new_path_index-thread;

	while(new_path_index<device_Paths_size[idx]+thread)device_Paths[new_path_index++]=-1;
	device_Paths_size[idx]=new_len;


	for(int i=0;i<30;i++)
	{printf("%d",device_Paths[i]);
	}
	printf("\n");

}




int main()
{


	int device_Paths_size[3];

	int dict[1251];
	memset(dict,-1,sizeof(dict));
	//    memset(device_Paths,-1,sizeof(device_Paths));

	memset(device_Paths_size,-1,sizeof(device_Paths_size));

	int device_Paths[]={1,2,3,4,2,3,5,-1,-1,-1,1,1,2,2,3,3,-1,-1,-1,-1,1,2,3,1,3,4,2,5,-1,-1};

	device_Paths_size[0]=7;
	device_Paths_size[1]=6;
	device_Paths_size[2]=8;

	repair(device_Paths,device_Paths_size,dict,0);
	//       memset(dict,-1,sizeof(dict));
	repair(device_Paths,device_Paths_size,dict,1);
	repair(device_Paths,device_Paths_size,dict,2);
}

