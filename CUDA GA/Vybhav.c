#include<stdio.h>
#include<stdlib.h>
int finder(int ind_val,int m,int* KSmallestElementsIndexArray,int op)
{// op=0 , finding element, op=1 inserting element;
	if(op==0)
	{
		int l=0,r=m;
		
		while(l<=r)
		{
			int mid=l+(r-l)/2;
			if(KSmallestElementsIndexArray[mid]==ind_val)
			return 0;
			if(KSmallestElementsIndexArray[mid]<ind_val)
			l=mid+1;
			else
			r=mid-1;
		}
		return 1;
	}
	else
	{
		if(m==0)
		{
			KSmallestElementsIndexArray[0]=ind_val;
		}
		else
		{
			int l = 0;
			int r = m;
			while(l < r)
			{
				int mid= l + (r - l)/2;
				
				if(KSmallestElementsIndexArray[mid] > ind_val) r = mid; // right could be the result
				else l = mid + 1; // m + 1 could be the result
			}
		
		// 1 element left at the end
		// post-processing
			int position= KSmallestElementsIndexArray[l] < ind_val ? l + 1: l;
			if(position==m+1)
				KSmallestElementsIndexArray[m]=ind_val;
			else 
			{
				for(int i=m;i>position;i--)
				{
					KSmallestElementsIndexArray[i]=KSmallestElementsIndexArray[i-1];
				}
				KSmallestElementsIndexArray[position]=ind_val;
			}

		}
	}
	return 1;
}



void Elimination(double* Array, int ArraySize, int* KSmallestElementsIndexArray, int K) 
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
	int m=0;
	while(m<K)
	{
		double mini=1e10;
		int index=-1;
		for(int i=0;i<ArraySize;i++)
		{
			if(Array[i]<=mini && finder(i,m,KSmallestElementsIndexArray,0))
			{
				index=i;
				mini=Array[i];
			}
		}
		
		if(index!=-1)
		{
			int y=finder(index,m,KSmallestElementsIndexArray,1);
		}
		m+=1;
	}
}
int main()
{
	int P = 5000;
	int ArraySize=(3*P)/2;
	double* array = (double*)calloc(sizeof(double),ArraySize);
	int K=ArraySize-P;
	int* KSmallestElementsIndexArray = (int*)calloc(sizeof(int),K);
	for(int testcase=0;testcase<5;testcase++)
	{
		printf("test case %d\n\n",testcase);
		for(int i=0;i<ArraySize;i++)
			array[i]=(double) rand() / (RAND_MAX);
/*		printf("Array input:\n");*/
/*		for(int i=0;i<ArraySize;i++)*/
/*			printf("%lf ",array[i]);*/
/*		printf("\n");*/
		Elimination(array,ArraySize,KSmallestElementsIndexArray,K);
/*		printf("Indeces output:\n");*/
/*		for(int i=0;i<K;i++)*/
/*			printf("%d ",KSmallestElementsIndexArray[i]);*/
/*		printf("\n\n");*/
	}
	
}
