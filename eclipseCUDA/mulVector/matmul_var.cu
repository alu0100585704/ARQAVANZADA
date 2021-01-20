/*
 ============================================================================
 Name        : matmul.cu
 Author      : jd
 Version     :
 Copyright   : Your copyright notice
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
 * CUDA kernels that compute product matrix
 */

__global__ void matMul1(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //NxM hilos distribuidos en bloques de (n x n) hilos
	unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned id_y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ( (id_x < d_col) && (id_y < d_fil) ){
        float acc = 0.0;
        //inicializa los punteros a las filas de A y C
        float *rowa = (float *)((char *)d_A + id_y * pitchA);
		float *rowc = (float *)((char *)d_C + id_y * pitchC);
		for (int k = 0; k < d_inner; k++){
			float *rowb = (float *)((char *)d_B + k * pitchB);
			acc = acc + (rowa[k] * rowb[id_x]);
		}
		rowc[id_x] = acc;
	}
}

__global__ void matMul1x(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //NxM hilos distribuidos en bloques de (n x n) hilos, con punteros
	unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned id_y = blockIdx.y*blockDim.y + threadIdx.y;

	if ( (id_x < d_col) && (id_y < d_fil) ) {
		float acc = 0.0;
		float *elemc = (float *)((char *)d_C + id_y * pitchC) + id_x;
		float *elema = (float *)((char *)d_A + id_y * pitchA); //Base de fila de A
		float *elemb = d_B + id_x; //Primer elemento de la columna de B
		for (int k = 0; k < d_inner; k++) {
			acc = acc + (*elema++) * (*elemb);
			elemb = (float *)((char *)elemb + pitchB); //Avanza el puntero por la columna de B
		}
		*elemc = acc;
	}
}

__global__ void matMul2(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
	//NxM bloques de I hilos
	unsigned i = blockIdx.x;
	unsigned j = blockIdx.y;
	unsigned k = threadIdx.x;

	__shared__ float acc;

	if (k == 0){
		acc = 0.0;
	}
	__syncthreads();
	float *el_a = (float *)((char *)d_A + i * pitchA) + k;
	float *el_b = (float *)((char *)d_B + k * pitchB) + j;
	float prod = (*el_a) * (*el_b);
	atomicAdd(&acc, prod);
	__syncthreads();
	if (k == 0){
	  float *el_c = (float *)((char *)d_C + i * pitchC) + j;
	  *el_c = acc;
	}
}

__global__ void matMul3(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
	//NxM bloques de I hilos
	unsigned i = blockIdx.x;
	unsigned j = blockIdx.y;
	unsigned k = threadIdx.x;

	float *el_c = (float *)((char *)d_C + i * pitchC) + j; //puntero a elemento a escribir
	if (k == 0){
		  *el_c = 0.0;
	}
	__syncthreads(); //espera a que se haya inicializado
	float *el_a = (float *)((char *)d_A + i * pitchA) + k;
	float *el_b = (float *)((char *)d_B + k * pitchB) + j;
	float prod = (*el_a) * (*el_b);
	atomicAdd(el_c, prod); //Todos los hilos escriben ordenadamente
}

__global__ void matMul4(float *d_A, float *d_B, float *d_C,
    unsigned d_fil, unsigned d_inner, unsigned d_col,
    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //M bloques de I hilos (columnas de B)
    unsigned j = blockIdx.x;
    unsigned k = threadIdx.x;
    __shared__ float acc;

    float el_b = *((float *)((char *)d_B + k * pitchB) + j); //Obtiene el elemento de B asociado a este hilo
    for (int i = 0; i < N; i++) {
        if (k == 0){
            acc = 0.0;
        }
        __syncthreads(); //espera a que se haya inicializado
        float el_a = *((float *)((char *)d_A + i * pitchA) + k);
        float *rowc = (float *)((char *)d_C + i * pitchC);
        atomicAdd(&acc, el_a * el_b); //Todos los hilos escriben ordenadamente
        __syncthreads(); //espera finalizar
        if (k == 1){
            rowc[j] = acc;
        }
    }
}

//Copia resultado de GPU al host y compara con resultado obtenido en host
int cmpMat(float *d_C, size_t pitchC, float *h_C, float *h_R, float tol)
{
    CUDA_CHECK_RETURN(cudaMemcpy2D (h_C, M * sizeof(float), d_C, pitchC, M * sizeof(float), N, cudaMemcpyDeviceToHost ));

    bool Ok = true;
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < M; j++){
            if (fabs(h_C[(i * M) + j] - h_R[(i * M) + j]) > tol){
                fprintf(stderr, "Result verification failed at element (%d, %d)! GPU=%f, CPU=%f\n", i, j, h_C[(i * M) + j], h_R[(i * M) + j]);
                Ok = false;
	        }
  	    }
    }
    return Ok;
}

int main(void)
{
    // Print the matrices dimensions
    printf("[Matrix multiplication of (%dx%d) X (%dx%d) matrices]\n", N, I, I, M);

    // Allocate the host input matrix A
    float *h_A = (float *)malloc(N*I*sizeof(float));
    // Allocate the host input matrix B
    float *h_B = (float *)malloc(I*M*sizeof(float));
    // Allocate the host output matrix C
    float *h_C = (float *)malloc(N*M*sizeof(float));
    // Allocate result verification matrix R
    float *h_R = (float *)malloc(N*M*sizeof(float));
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_R == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
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

    //Compute result matrix
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < M; j++){
           float acc = 0.0;
           for (int k = 0; k < I; k++){
    		acc = acc + (h_A[(i * I) + k] * h_B[(k * M) + j]);
           }
           h_R[(i * M) + j] = acc;
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
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_A, pitchA, h_A, I * sizeof(float), I * sizeof(float), N, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_B, pitchB, h_B, M * sizeof(float), M * sizeof(float), I, cudaMemcpyHostToDevice ));

    // Launch the Matrix product CUDA Kernel
    dim3 threadsPerBlock1(16,16);
    dim3 blocksPerGrid1( (M + threadsPerBlock1.x - 1) / threadsPerBlock1.x , (N + threadsPerBlock1.y - 1) / threadsPerBlock1.y );
    printf("CUDA kernel 1 launch with (%d, %d) blocks of (%d, %d) threads\n", blocksPerGrid1.x, blocksPerGrid1.y, threadsPerBlock1.x, threadsPerBlock1.y);
    matMul1<<<blocksPerGrid1, threadsPerBlock1>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    // Launch the Matrix product CUDA Kernel
    dim3 threadsPerBlock1x(16,16);
    dim3 blocksPerGrid1x( (M + threadsPerBlock1x.x - 1) / threadsPerBlock1x.x , (N + threadsPerBlock1x.y - 1) / threadsPerBlock1x.y );
    printf("CUDA kernel 1x launch with (%d, %d) blocks of (%d, %d) threads\n", blocksPerGrid1x.x, blocksPerGrid1x.y, threadsPerBlock1x.x, threadsPerBlock1x.y);
    matMul1x<<<blocksPerGrid1x, threadsPerBlock1x>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    int threadsPerBlock2 = I;
    dim3 blocksPerGrid2(N, M);
    printf("CUDA kernel 2 launch with (%d, %d) blocks of %d threads\n", blocksPerGrid2.x, blocksPerGrid2.y, threadsPerBlock2);
    matMul2<<<blocksPerGrid2, threadsPerBlock2>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    int threadsPerBlock3 = I;
    dim3 blocksPerGrid3(N, M);
    printf("CUDA kernel 3 launch with (%d, %d) blocks of %d threads\n", blocksPerGrid3.x, blocksPerGrid3.y, threadsPerBlock3);
    matMul3<<<blocksPerGrid3, threadsPerBlock3>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    int threadsPerBlock4 = I;
    int blocksPerGrid4 = M;
    printf("CUDA kernel 4 launch with %d blocks of %d threads\n", blocksPerGrid4, threadsPerBlock4);
    matMul4<<<blocksPerGrid4, threadsPerBlock4>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    // Free device global memory
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_R);

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

