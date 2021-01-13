#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


/**
 * CUDA Kernel Device code
 * Calcula el histograma de un vector pasado. 
 * Versión directa con operaciones atómicas a vector H en memoria de video
 */
__global__ void
histogram(int *V, int * H, int numElementsV, int numElementsH)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	
if (i < numElementsV)
  {
	int index = V[i] % numElementsH;
	atomicAdd((H +index), 1); 
  }
	
}

/**
 * CUDA Kernel Device code
 * Calcula el histograma de un vector pasado.
 * Versión directa con operaciones atómicas a memoria de video, pero con un histograma por bloque.
 * Se sigue utilizando un vector H, pero su tamaño es numElementsH * número de bloques.
 * O sea, hay un histograma de tamaño numElementsH por cada bloque y su ubicación en memoria es como vector, un histograma seguido del otro
 */
__global__ void
histogramByBlock(int *V, int * H, int numElementsV, int numElementsH)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	
if (i < numElementsV)
  {	
	int index = (V[i] % numElementsH) + (blockIdx.x * numElementsH); ///me posiciono en histograma asociado a este bloque y en la posición correspondiente
	atomicAdd((H +index), 1); 
  }
	
}

/**
 * CUDA Kernel Device code
 * Calcula el histograma de un vector pasado.
 * Versión directa con operaciones atómicas a memoria compartida por un bloque y un solo acceso atómico final, tras sincronización de los hilos
 * con el resultado.
 * a memorida de video.
 */
__global__ void
histogramShared(int *V, int * H, int numElementsV, int numElementsH)
{
__shared__ int acc[50];  ///tamaño máximo del vector H.Ver si se puede optimizar para  crear en tiempo de ejecución el vector dentro del kernel.

 int i = blockDim.x * blockIdx.x + threadIdx.x;
	
if (i < numElementsV)
  {
	int index = V[i] % numElementsH;
	atomicAdd((H +index), 1); 
  }
	
}

/**
 * CUDA Kernel Device code
 * Calcula el histograma de un vector pasado.
 * Versión directa con operaciones atómicas a memoria compartida por un bloque y un solo acceso atómico final, tras sincronización de los hilos
 * con el resultado.
 * a memorida de video.
 * Devuelve un puntero hacia el el vector del histograma ya calculado
 */
int * calculateHistogramByGpu(int * vector,int numElementsV, int numElementsH, bool byBlock, int threadsPerBlock)
{
	size_t sizeH,sizeV;
	int *h_H = NULL;
	int *d_V = NULL;
	int *d_H = NULL;	
    cudaError_t err = cudaSuccess;
	
	sizeV =  numElementsV * sizeof(int);

	if (threadsPerBlock >1024)
	    threadsPerBlock = 1024;  ///para no sobrepasar el límite de bloque. Realmente este valor se deberá de obtener de la función CUDA adecuada, puesto que podría no ser
							    		
    int blocksPerGrid = (numElementsV + threadsPerBlock - 1) / threadsPerBlock;
	
 if (byBlock)
	///si creo la versión de un histograma por bloque	
	sizeH = numElementsH * blocksPerGrid * sizeof(int) ;
 else
   ///si creo la versión de un histograma único para todo el vector.   
	sizeH = numElementsH * sizeof(int);

 	h_H = (int *)malloc(sizeH);
	
    // Verify that allocations succeeded
    if (h_H == NULL)
    {
        fprintf(stderr, "Failed to allocate host vector Histograma!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vector H a cero
    printf("\nVector Histrograma Inicializado con 0:");
    for (int i = 0; i < (sizeH / sizeof(int)); ++i)    
		h_H[i] = 0;
	
        
    // Allocate the device vector V       
    err = cudaMalloc((void **)&d_V, sizeV);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

// Allocate the device vector H    

    err = cudaMalloc((void **)&d_H, sizeH);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector H (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the host input vectors V in host memory to the device input vectors in
    // device memory
    printf("\nCopy input data from the host memory vector V to the CUDA device");
    err = cudaMemcpy(d_V, vector, sizeV, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector V from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors H in host memory to the device input vectors in
    // device memory
    printf("\nCopy input data from the host memory vector H to the CUDA device");
    err = cudaMemcpy(d_H, h_H, sizeH, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector H from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Launch the Vector Add CUDA Kernel   
	if (byBlock)	
	{
	    printf("\nCUDA kernel - histogramByBlock - launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);       
		histogramByBlock<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_H, numElementsV, numElementsH);

	}
	else
	{
		printf("\nCUDA kernel -histogram- launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);       
		histogram<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_H, numElementsV, numElementsH);
	
	}
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    // Copy the device result vector V in device memory to the host result vector
    // in host memory.
    printf("\nCopy output data from the CUDA device vector V to the host memory");
    err = cudaMemcpy(vector, d_V, sizeV, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector V from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the device result vector H in device memory to the host result vector
    // in host memory.
    printf("\nCopy output data from the CUDA device vector H to the host memory");
    err = cudaMemcpy(h_H, d_H, sizeH, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector H from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Free device global memory
    err = cudaFree(d_V);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_H);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


return h_H;
}

/**
 * Calcula el histograma de un vector pasado.
 * Versión directa con operaciones atómicas a memoria compartida por un bloque y un solo acceso atómico final, tras sincronización de los hilos
 * con el resultado.
 * a memorida de video.
 */
/*int * calculateHistogramByCpu(int * vector, int sizeHistogram, bool byBlock)
{
int numElementsH = 8;	
return 
}*/

/**
 * Host main routine
 */
int main(void)
{

	cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElementsV = 50000000;
	int numElementsH = 8;
	int threadsPerBlock = 1024;
	    
    printf("\nVector V de %d elementos", numElementsV);
    // Allocate the host input vector V
    int *h_V = (int *)malloc(numElementsV * sizeof(int));

    // Verify that allocations succeeded
    if (h_V == NULL)
    {
        fprintf(stderr, "Failed to allocate host vector V!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vector V
    printf("\nVector V Inicializado con :");

    for (int i = 0; i < numElementsV; ++i)    
	 {
		//	h_V[i] = rand();  ///(float)RAND_MAX;
			h_V[i] = i;
		if  (numElementsV<1025)
			printf("\n[%d]", h_V[i]); ///solo muestro por pantalla si es menor o igual de 1024
     }	
	 

    int * h_H = calculateHistogramByGpu(h_V, numElementsV, numElementsH, false, threadsPerBlock);

///Show Vector H
    printf("\nResultado Vector Histograma  :");
    for (int i = 0; i < numElementsH; ++i)    
		 {			
			printf("\n[%d]", h_H[i]);
		}	

free(h_H);

     h_H = calculateHistogramByGpu(h_V, numElementsV, numElementsH, true, threadsPerBlock);

///Show Vector H
    printf("\nResultado Vector Histograma  :");
    for (int i = 0; i < numElementsH; ++i)    
		 {			
			printf("\n[%d]", h_H[i]);
		}	

free(h_H);





    // Free host memory
    free(h_V);
	

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
