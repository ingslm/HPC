#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 10



int * sumar(int *A, int *B, int *C){
   
   for(int i=0;i<TAM;i++){
     C[i]= A[i]+B[i];
     printf("%d",C[i]);
   }
   return C;
 }
    

int main(){
	int n; //longitud del vector
	int * A;
	int * B;
	int * C;
  	n=TAM*sizeof(int);

  	clock_t t;
  	
  	
  	

	A = (int*)malloc( n );
	B = (int*)malloc( n);
	C = (int*)malloc( n);

	for(int i=0;i<TAM;i++){
		A[i]=rand() % 10 ;
    	printf("%d",A[i]);
		B[i]=rand() % 10;
    	printf("%d\n",B[i]);
	}
	
	
  	


  	t = clock();
  	
  	free(A);
  	free(B);
  	free(C);

	
    sumar(A,B,C);

  	t = clock() - t;
  	printf ("\nIt took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

	return 0;
}
