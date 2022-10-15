#include<stdio.h>
#include<stdlib.h>
void Elimination(int NumCols, int Matrix[][NumCols], double* Array, int* AuxillaryArray, int NumRows) //Must be done INPLACE
{
	//PathMatrix is the matrix and Fitness is the array 
	;
}
int main()
{
	int P = 4;
	int NumRows=(3*P)/2;
	int NumCols=6;
	int Matrix[NumRows][NumCols];
	double Array[NumRows];
	int AuxillaryArray[NumRows];
	for(int i=0;i<NumRows;i++)
	{
		for(int j=0;j<NumCols;j++)
			Matrix[i][j]=abs(rand())%100;
		Array[i]=((double) rand() / (RAND_MAX));
		AuxillaryArray[i]=((int) rand()%300);
	}
	printf("\n");
	printf("ORIGINAL \nPath\t\t\tArray\t\t\tAux\n");
	for(int i=0;i<NumRows;i++)
	{
		for(int j=0;j<NumCols;j++)
			printf("%d ",Matrix[i][j]);
		printf("\t%lf\t",Array[i]);
		printf("\t%d\n",AuxillaryArray[i]);
	}
	printf("\n");
	printf("\n");
	Elimination(NumCols,Matrix,Array,AuxillaryArray,NumRows);
	printf("OUTPUT \nPath\t\t\tArray\t\t\tAux\n");
	for(int i=0;i<NumRows;i++)
	{
		for(int j=0;j<NumCols;j++)
			printf("%d ",Matrix[i][j]);
		printf("\t%lf\t",Array[i]);
		printf("\t%d\n",AuxillaryArray[i]);
	}
}
