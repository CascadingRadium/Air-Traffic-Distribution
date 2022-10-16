#include<stdio.h>
#include<stdlib.h>
struct Pair{
    double data;
    int index;
}typedef Pair;
void heapify(Pair* heap,int n)
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

void deleteNode(Pair* heap,int currentSize)
{
    Pair lastElement = heap[currentSize - 1];
 
    heap[0] = lastElement;
 
    currentSize--;

    heapify(heap, currentSize);
}

void Elimination(double* Array, int ArraySize, int* KSmallestElementsIndexArray, Pair* heap, int K) 
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
	int P = 9000;
	int ArraySize=(3*P)/2;
	double* array = (double*)calloc(sizeof(double),ArraySize);
	int K=ArraySize-P;
	int* KSmallestElementsIndexArray = (int*)calloc(sizeof(int),K);
	Pair* heap = (Pair*)malloc(sizeof(Pair)*ArraySize);
	for(int testcase=0;testcase<100;testcase++)
	{
		printf("test case %d\n\n",testcase);
		for(int i=0;i<ArraySize;i++)
			array[i]=(double) rand() / (RAND_MAX);
		printf("Array input:\n");
/*		for(int i=0;i<ArraySize;i++)*/
/*			printf("%lf ",array[i]);*/
		printf("\n");
		Elimination(array,ArraySize,KSmallestElementsIndexArray,heap,K);
		printf("Indeces output:\n");
		for(int i=0;i<K-1;i++)
		{
			//printf("%d ",KSmallestElementsIndexArray[i]);
			if(KSmallestElementsIndexArray[i]==KSmallestElementsIndexArray[i+1])
			    printf("\n\nCAUGHT\n\n");
	}
		printf("\n\n");
	}
	
}
