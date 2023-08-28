
//transpose.cu
#include <stdio.h>
#include "gputimer.h"
// #include "utils.h"

const int N = 1024;	// matrix size will be NxN
const int K = 16;

int compare_matrices(float *gpu, float *ref, int N)
{
        int result = 0;
        for(int j=0; j < N; j++)
        for(int i=0; i < N; i++)
                if (ref[i + j*N] != gpu[i + j*N])
                   {result = 1;}
        return result;
}

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat, int N)
{
        for(int j=0; j < N * N; j++)
                mat[j] = (float) j;
}

void 
transpose_CPU(float in[], float out[])
{
	for(int j=0; j < N; j++)
    	    for(int i=0; i < N; i++)
      		out[j + i*N] = in[i + j*N]; // implements flip out(j,i) = in(i,j)
}

// to be launched on a single thread
__global__ void 
transpose_serial(float in[], float out[])
{
	for(int j=0; j < N; j++)
		for(int i=0; i < N; i++)
			out[j + i*N] = in[i + j*N]; 
}

// to be launched with one thread per row of output matrix
__global__ void 
transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x + blockDim.x * blockIdx.y;

	for(int j=0; j < N; j++)
		out[j + i*N] = in[i + j*N]; 
}


// Assignment #3: Write two tiled versions of transpose -- One using shared memory. 
// To be launched with one thread per element, in KxK threadblocks.
// You will determine for each thread (x,y) in tile the element (i,j) of global output matrix. 

__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{
	// init. i and j variables, and assign them to the threatIdx of 
	// x and y added to their unique block ID times K (creating KxK 
	// threadblocks).
	int i = threadIdx.x + (blockIdx.x * K);
	int j = threadIdx.y + (blockIdx.y * K);	

	// Loop isn't required since each element in the matrix has a 
	// threadblock of it's own.
	out[j + i*N] = in[i + j*N];	
}

__global__ void 
transpose_parallel_per_element_tiled_shared(float in[], float out[])
{
	// init. a shared "tile" with the same KxK size as the input matrix
	__shared__ int tile[K][K];

	// init. variables i and j, for the input x and y coords of each 
	// element in our input matrix, seen in transpose_parallel_per_element
	int i = blockIdx.x * K;
	int j = blockIdx.y * K;

	// Store thread id in x and y for easy referencing
	int x = threadIdx.x;
	int y = threadIdx.y;	

	// Transpose matrix into the shared memory tile 
	tile[y][x] = in[(i+x) + (j+y)*N];
	
	// Synchronize all threads (to compile a final tile)
	__syncthreads();

	// Once tile matrix is fully built, write it to global out[]
	out[(j+x) + (i+y)*N] = tile[x][y];
}

int main(int argc, char **argv)
{
	int numbytes = N * N * sizeof(float);
	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	float *gold = (float *) malloc(numbytes);
	fill_matrix(in, N);
	transpose_CPU(in, gold);

	float *d_in, *d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;
        timer.Start();
	transpose_serial<<<1,1>>>(d_in, d_out);
	timer.Stop();
        for (int i=0; i < N*N; ++i){out[i] = 0.0;}
        cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_serial: %g ms.\nVerifying ...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

   
        cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice); //clean d_out
        timer.Start();
	transpose_parallel_per_row<<<1,N>>>(d_in, d_out);
	timer.Stop();
        for (int i=0; i < N*N; ++i){out[i] = 0.0;}  //clean out
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_row: %g ms.\nVerifying ...%s\n", 
		    timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

        cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice); //clean d_out
        // Tiled versions
        // const int K= 16;
        dim3 blocks_tiled(N/K,N/K);
	dim3 threads_tiled(K,K);
	timer.Start();
	transpose_parallel_per_element_tiled<<<blocks_tiled,threads_tiled>>>(d_in, d_out);
	timer.Stop();
        for (int i=0; i < N*N; ++i){out[i] = 0.0;}
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

        cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice); //clean d_out
        dim3 blocks_tiled_sh(N/K,N/K);
	dim3 threads_tiled_sh(K,K);
        timer.Start();
	transpose_parallel_per_element_tiled_shared<<<blocks_tiled_sh,threads_tiled_sh>>>(d_in, d_out);
	timer.Stop();
        for (int i=0; i < N*N; ++i){out[i] = 0.0;}
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled_shared %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

	cudaFree(d_in);
	cudaFree(d_out);
}
