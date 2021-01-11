/*
 ============================================================================
 Name        : matmul.cu
 Description : CUDA compute matrix multip
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

#define N (200)
#define M (300)
#define I (100)

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes product matrix
 */

__global__ void matMul(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{


}



int main(void)
{
    // Print the vector length to be used, and compute its size
    printf("[Matrix multiplication of (%dx%d) X (%dx%d) matrices]\n", N, I, I, M);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(N*I*sizeof(float));
    // Allocate the host input vector B
    float *h_B = (float *)malloc(I*M*sizeof(float));
    // Allocate the host output vector C
    float *h_C = (float *)malloc(N*M*sizeof(float));
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host matrix A
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < I; j++){
    		h_A[i*I+j] = rand()/(float)RAND_MAX;
    	}
    }
    // Initialize the host matrix B
    for (int i = 0; i < I; i++){
        for (int j = 0; j < M; j++){
        	h_B[i*M+j] = rand()/(float)RAND_MAX;
        }
    }

    // Allocate the device input matrix A
    float *d_A = NULL;
    size_t pitchA;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_A, &pitchA, I * sizeof(float), N));
    //printf("[Pitch of A is %d, Width is %d, Width in bytes is %d and height is %d]\n", pitchA, I, I*sizeof(float), N);

    // Allocate the device input matrix B
    float *d_B = NULL;
    size_t pitchB;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_B, &pitchB, M * sizeof(float), I));

    // Allocate the device output matrix C
    float *d_C = NULL;
    size_t pitchC;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_C, &pitchC, M * sizeof(float), N));

    // Copy the host input matrices A and B in host memory to the device input matrices in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_A, pitchA, h_A, I * sizeof(float), I * sizeof(float), N, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_B, pitchB, h_B, M * sizeof(float), M * sizeof(float), I, cudaMemcpyHostToDevice ));

    // Launch the Matrix product CUDA Kernel
    dim3 threadsPerBlock1(,);
    dim3 blocksPerGrid1(,);
    //printf("CUDA kernel 1 launch with (%d, %d) blocks of (%d, %d) threads\n", blocksPerGrid1.x, blocksPerGrid1.y, threadsPerBlock1.x, threadsPerBlock1.y);
    matMul<<<blocksPerGrid1, threadsPerBlock1>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());

    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    CUDA_CHECK_RETURN(cudaMemcpy2D (h_C, M * sizeof(float), d_C, pitchC, M * sizeof(float), N, cudaMemcpyDeviceToHost ));

    // Verify that the result matrix is correct
    bool Ok = true;
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < M; j++){
    		float acc = 0.0;
    		for (int k = 0; k < I; k++){
    			acc = acc + (h_A[(i * I) + k] * h_B[(k * M) + j]);
    		}
    		if (fabs(acc - h_C[(i * M) + j]) > 1.0e-3){
    			fprintf(stderr, "Result verification failed at element (%d, %d)! GPU=%f, CPU=%f\n", i, j, h_C[(i * M) + j], acc);
    			Ok = false;
    		    //exit(EXIT_FAILURE);
    		}
    	}
    }
    if (Ok) printf("Test PASSED\n");

    // Free device global memory
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return EXIT_SUCCESS;

}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}

