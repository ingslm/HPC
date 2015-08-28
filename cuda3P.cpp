#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 10


__global__ void vecAddGPU(int *A, int *B, int *C){
	
	int tid = threadIdx.x;
	if(tid < TAM){
		C[tid] = A[tid]+B[tid];
		printf("%d",C[tid]);
	}
}

int * sumar(int *A, int *B, int *C, int n){
   
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
  	//int f;
  	
  	

	A = (int*)malloc( n );
	B = (int*)malloc( n);
	C = (int*)malloc( n);

	for(int i=0;i<TAM;i++){
		A[i]=rand() % 10 ;
    	printf("%d",A[i]);
		B[i]=rand() % 10;
    	printf("%d\n",B[i]);
	}
	
	
  	int *d_a, *d_b, *d_c;


  	cudaMalloc( (void **) &d_a,n );
  	cudaMalloc( (void **) &d_b,n );
  	cudaMalloc( (void **) &d_c,n);
  	t = clock();
  	cudaMemcpy(d_a,A,n,cudaMemcpyHostToDevice);
  	cudaMemcpy(d_b,B,n,cudaMemcpyHostToDevice);
  	vecAdd<<<1,1024>>>(d_a,d_b,d_c);
  	cudaMemcpy(C,d_c,n,cudaMemcpyDeviceToHost);
  	cudaFree(d_a);
  	cudaFree(d_b);
  	cudaFree(d_c);
  	free(A);
  	free(B);
  	free(C);

	//vecAddGPU(A,B,C);
  //	sumar(A,B,C,n);

  	t = clock() - t;
  	printf ("\nIt took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

	return 0;
}
