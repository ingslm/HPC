#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 2048
#define THREAD 1024

int blocks(int tam){
  int n = (tam/THREAD);
  if(n<=1)
    return 1;
  else
    return n;
}


__global__ void vecAdd(int *A, int *B, int *C, int n){
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n){
		C[tid] = A[tid]+B[tid];
		}
}

int * sumar(int *A, int *B, int *C, int n){
   
   for(int i=0;i<n;i++){
     C[i]= A[i]+B[i];
   }
   return C;
 }
    

int main(){
  int tams[]={512,1024,3000,10000, 500000};
	int n; //longitud del vector
  for(int j=0;j<4;j++){
	int * A;
	int * B;
	int * C;
  	n=tams[j]*sizeof(int);

  	clock_t t,s;
  	//int f;
  	
  	
	printf("hola");
	A = (int*)malloc( n );
	B = (int*)malloc( n);
	C = (int*)malloc( n);

	for(int i=0;i<tams[j];i++){
		A[i]=rand() % 10 ;
		B[i]=rand() % 10;
    	
	}
	
	
  	int *d_a, *d_b, *d_c;


  	cudaMalloc( (void **) &d_a,n );
  	cudaMalloc( (void **) &d_b,n );
  	cudaMalloc( (void **) &d_c,n);
  	t = clock();
  	cudaMemcpy(d_a,A,n,cudaMemcpyHostToDevice);
  	cudaMemcpy(d_b,B,n,cudaMemcpyHostToDevice);
    int dimGrid = blocks(tams[j]);
    printf("%d",dimGrid);
    int tam = tams[j];
  	vecAdd<<< dimGrid,THREAD>>>(d_a,d_b,d_c,tam);
  	cudaMemcpy(C,d_c,n,cudaMemcpyDeviceToHost);
  	cudaFree(d_a);
  	cudaFree(d_b);
  	cudaFree(d_c);
  	free(A);
  	free(B);
  	free(C);

	//vecAddGPU(A,B,C);
    
  	t = clock() - t;
  	printf ("\nIt took me in GPU with TAM %d is (%f seconds).\n",tams[j],((float)t)/CLOCKS_PER_SEC);
  
    s = clock();
  	sumar(A,B,C,tams[j]);
    s = clock() - s;
    printf ("\nIt took me in CPU with TAM %d is (%f seconds).\n", tams[j],((float)s)/CLOCKS_PER_SEC);
  }
	return 0;
}
