#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 * Incrementa cada componente del vector A con numElements elementos
 */
__global__ void
vectorInc(float *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        A[i]++;

    }
}

///separo el kernel en dos, uno de lectura y el otro de escritura, para que sea seguro la rotación del vector
///y aprovechando el paralelismo. ASí no se tiene que asignar memoria origen y destino, y aprovechamos la misma
///del vector

__global__ void
invVectorRead(float *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a;

    if (i < numElements)
    {   a = A[(numElements-1) -i];
    	__syncthreads();
    	A[i] = a;
    }

}

__global__ void
invVectorWrite(float *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a;
    if (i < numElements)
    {
         a = A[(numElements-1) -i];

    }
}


/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 20;
    size_t size = numElements * sizeof(float);
    printf("[Vector increment of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);


    // Verify that allocations succeeded
    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vector!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vector
    printf("\nVector Inicializado con \n");

    for (int i = 0; i < numElements; ++i)
    {
    		h_A[i] = i;
        	printf("[%f]", h_A[i]);
    }


    // Allocate the device vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A in host memory to the device input vectors in
    // device memory
    printf("\nCopy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 1;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("\nCUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cudaEvent_t start,stop;
    float time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    invVectorRead<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    err = cudaGetLastError();

    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("\nTime elapsed by kernel: %f\n", time_ms);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nVector invertido\n");
    // Initialize the host input vector
    for (int i = 0; i < numElements; ++i)
        	printf("[%f]", h_A[i]);


    printf("\nVector invertido desde memoria de dispositivo %p \n",d_A);

    // Initialize the host input vector
    for (int i = 0; i < numElements; ++i)
        	printf("[%f]", d_A[i]);
    // Free device global memory

    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nDone\n");
    return 0;
}
