//#include<stdio.h>
//#include<stdlib.h>
//void Elimination(double* Array, int ArraySize, int* KSmallestElementsIndexArray, int K) 
//{
//	/* Requirements:
//	   Function should give me the indices of the K lowest/smallest elements in the array of double (Array) of size ArraySize. The result (K indices of K least elements must be stored in the array KSmallestElementsIndexArray of size K). -> Use the heap of size k provided for the heap method. (Optional) Assume heap is always filled with junk when function is called.
//	   1.	No recursion
//	   2.  No sorting/using any inbuilt function
//	   3.	Code in C only.
//	   4.	Do not use dynamically determined extra epace ex int a[n] where n is runtime dependent is not allowed in cuda
//	   5.	O(Nlogk) time and O(k) Space. (optional - based on choice)
//	   6.	The Result array must be sorted in ascending order.
//	   7.	Array is immutable
//	   Thanks
//	   Example 0.7 0.6 0.2 0.3 0.1  ->input K = 3
//	   output -> 4 2 3 -> Final Output - 2 3 4
//	   Ranges -> ArraySize = 6000
//	   K -> 2000
//	 */
//	int m=0;
//	int l=0;
//	int r=m;
//	int mid=0;
//	int position;
//	int i=0;
//	double mini=1e10;
//	int index=-1;
//	bool ret=1;
//	while(m<K)
//	{
//		mini=1e10;
//		index=-1;
//		for(i=0;i<ArraySize;i++)
//		{
//			l=0;
//			r=m;
//			ret=1;
//			while(l<=r)
//			{
//				mid=l+(r-l)/2;
//				if(KSmallestElementsIndexArray[mid]==i)
//				{
//					ret=0;
//					break;
//				}
//				if(KSmallestElementsIndexArray[mid]<i)
//					l=mid+1;
//				else
//					r=mid-1;
//			}
//			
//			
//			if(Array[i]<=mini && ret)
//			{
//				index=i;
//				mini=Array[i];
//			}
//		}
//		
//		
//		if(index!=-1)
//		{
//			if(m==0)
//				KSmallestElementsIndexArray[0]=index;
//			else
//			{
//				l = 0;
//				r = m;
//				while(l < r)
//				{
//					mid= l + (r - l)/2;
//					if(KSmallestElementsIndexArray[mid] > index) 
//						r = mid; // right could be the result
//					else 
//						l = mid + 1; // m + 1 could be the result
//				}
//				// 1 element left at the end
//				// post-processing
//				if(KSmallestElementsIndexArray[l] < index)
//					position=l+1;
//				else
//					position=l;
//				if(position==m+1)
//					KSmallestElementsIndexArray[m]=index;
//				else 
//				{
//					for(i=m;i>position;i--)
//					{
//						KSmallestElementsIndexArray[i]=KSmallestElementsIndexArray[i-1];
//					}
//					KSmallestElementsIndexArray[position]=index;
//				}
//			}
//		}
//		m+=1;
//	}
//}
////int finder(int ind_val,int m,int* KSmallestElementsIndexArray,int op)
//////{ op=0 , finding element, op=1 inserting element;
////	if(op==0)
////	{
////		int l=0,r=m;
////		while(l<=r)
////		{
////			int mid=l+(r-l)/2;
////			if(KSmallestElementsIndexArray[mid]==ind_val)
////				return 0;
////			if(KSmallestElementsIndexArray[mid]<ind_val)
////				l=mid+1;
////			else
////				r=mid-1;
////		}
////		return 1;
////	}
////	else
////	{
////		if(m==0)
////		{
////			KSmallestElementsIndexArray[0]=ind_val;
////		}
////		else
////		{
////			int l = 0;
////			int r = m;
////			while(l < r)
////			{
////				int mid= l + (r - l)/2;
////				if(KSmallestElementsIndexArray[mid] > ind_val) r = mid;  right could be the result
////				else l = mid + 1;  m + 1 could be the result
////			}
////			 1 element left at the end
////			 post-processing
////			int position= KSmallestElementsIndexArray[l] < ind_val ? l + 1: l;
////			if(position==m+1)
////				KSmallestElementsIndexArray[m]=ind_val;
////			else 
////			{
////				for(int i=m;i>position;i--)
////				{
////					KSmallestElementsIndexArray[i]=KSmallestElementsIndexArray[i-1];
////				}
////				KSmallestElementsIndexArray[position]=ind_val;
////			}
////		}
////	}
////	return 1;
////}
////void Elimination(double* Array, int ArraySize, int* KSmallestElementsIndexArray, int K) 
////{
////	/* Requirements:
////	   Function should give me the indices of the K lowest/smallest elements in the array of double (Array) of size ArraySize. The result (K indices of K least elements must be stored in the array KSmallestElementsIndexArray of size K). -> Use the heap of size k provided for the heap method. (Optional) Assume heap is always filled with junk when function is called.
////	   1.	No recursion
////	   2.  No sorting/using any inbuilt function
////	   3.	Code in C only.
////	   4.	Do not use dynamically determined extra epace ex int a[n] where n is runtime dependent is not allowed in cuda
////	   5.	O(Nlogk) time and O(k) Space. (optional - based on choice)
////	   6.	The Result array must be sorted in ascending order.
////	   7.	Array is immutable
////	   Thanks
////	   Example 0.7 0.6 0.2 0.3 0.1  ->input K = 3
////	   output -> 4 2 3 -> Final Output - 2 3 4
////	   Ranges -> ArraySize = 6000
////	   K -> 2000
////	 */
////	int m=0;
////	while(m<K)
////	{
////		double mini=1e10;
////		int index=-1;
////		for(int i=0;i<ArraySize;i++)
////		{
////			if(Array[i]<=mini && finder(i,m,KSmallestElementsIndexArray,0))
////			{
////				index=i;
////				mini=Array[i];
////			}
////		}
////		if(index!=-1)
////		{
////			int y=finder(index,m,KSmallestElementsIndexArray,1);
////		}
////		m+=1;
////	}
////}
#include<stdio.h>
#include<stdlib.h>
struct Pair{
	double data;
	int index;
}typedef Pair;
__device__ void heapify(Pair* heap,int n)
{
	int parent = (n -1)/ 2;
	for (int i = parent; i >= 0; i--){
		int k = i; 
		Pair v = heap[k]; 
		int isheap = 0; 
		while(!isheap && 2 * k <= n){
			int j = 2 * k; 
			if (j < n) 
			{ 
				if (heap[j].data > heap[j + 1].data)
					j = j + 1; 
			}
			if (v.data <= heap[j].data)
				isheap = 1; 
			else{
				heap[k] = heap[j]; 
				k = j; 
			}
		}
		heap[k]= v; 
	}
}
__device__ void deleteNode(Pair* heap,int currentSize)
{
	Pair lastElement = heap[currentSize - 1];
	heap[0] = lastElement;
	currentSize--;
	heapify(heap, currentSize);
}
__global__ void Elimination(double* Array, int ArraySize, int* KSmallestElementsIndexArray, Pair* heap, int K) 
{
	/* Requirements:
	   Function should give me the indices of the K lowest/smallest elements in the array of double (Array) of size ArraySize. The result (K indices of K least elements must be stored in the array KSmallestElementsIndexArray of size K). -> Use the heap of size k provided for the heap method. (Optional) Assume heap is always filled with junk when function is called.
	   1.	No recursion
	   2.  No sorting/using any inbuilt function
	   3.	Code in C only.
	   4.	Do not use dynamically determined extra epace ex int a[n] where n is runtime dependent is not allowed in cuda
	   5.	O(Nlogk) time and O(k) Space. (optional - based on choice)
	   6.	The Result array must be sorted in ascending order.
	   7.	Array is immutable
	   Thanks
	   Example 0.7 0.6 0.2 0.3 0.1  ->input K = 3
	   output -> 4 2 3 -> Final Output - 2 3 4
	   Ranges -> ArraySize = 6000
	   K -> 2000
	 */
	for(int i=0;i<ArraySize;i++)
	{
		Pair x;
		x.data=Array[i];
		x.index=i;
		heap[i]=x;
	}
	heapify(heap,ArraySize-1);
	int currentSize=ArraySize;
	int n=0;
	while(K--)
	{
		int i =n -1;
		int item=heap[0].index;
		while(item< KSmallestElementsIndexArray[i] && i>=0)
		{
			KSmallestElementsIndexArray[i+1] = KSmallestElementsIndexArray[i];
			i--;
		}
		KSmallestElementsIndexArray[i+1] = item;
		n++;
		deleteNode(heap,currentSize);
		currentSize--;
	}
}
int main()
{
	srand(time(NULL));
	int P = 5000;
	int ArraySize=(3*P)/2;
	double* host_array = (double*)calloc(sizeof(double),ArraySize);
	for(int i=0;i<ArraySize;i++)
		host_array[i]=(double) rand() / (RAND_MAX);
	double* array;
	cudaMalloc((void**)&array,sizeof(double)*ArraySize);
	cudaMemcpy(array,host_array,sizeof(double)*ArraySize,cudaMemcpyHostToDevice);
	int K=ArraySize-P;
	int* KSmallestElementsIndexArray;
	cudaMalloc((void**)&KSmallestElementsIndexArray,sizeof(int)*K);
	int* host_KSmallestElementsIndexArray =(int*)calloc(sizeof(int),K);
	Pair* heap; 
	cudaMalloc((void**)&heap,(sizeof(Pair)*ArraySize));
	for(int testcase=0;testcase<1;testcase++)
	{
			for(int i=0;i<ArraySize;i++)
				host_array[i]=(double) rand() / (RAND_MAX);
			cudaMemcpy(array,host_array,sizeof(double)*ArraySize,cudaMemcpyHostToDevice);	
//				printf("Array input:\n");
//				for(int i=0;i<ArraySize;i++)
//					printf("%lf ",host_array[i]);
//				printf("\n");
		Elimination<<<1,1>>>(array,ArraySize,KSmallestElementsIndexArray,heap,K);
//				cudaDeviceSynchronize();
//				cudaMemcpy(host_KSmallestElementsIndexArray,KSmallestElementsIndexArray,sizeof(int)*K,cudaMemcpyDeviceToHost);
//				printf("Indeces output:\n");
//				for(int i=0;i<K;i++)
//					printf("%d ",host_KSmallestElementsIndexArray[i]);
				printf("\n\n");
	}
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess) 
		printf("CUDA error: %s\n",cudaGetErrorString(err)); 
	cudaDeviceSynchronize();

}





