/*
 ============================================================================
 Name        : sumvec_gsync.cu
 Ejemplo de suma de elementos de un vector con operaciones atómicas, medida
 de tiempos con clock64() y uso de variables ubicadas directamente en memoria
 global de la GPU. No hay garantías de corrección si se inicializa la suma así
 y hay más de un bloque de hilos por la sincronización entre bloques
 Variación para evitar esto último usando cooperative groups
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

//Preparamos namespace, o alternativamente ponemos using ... y pasamos del prefijo
namespace cg = cooperative_groups; 

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

//Num. de elementos a sumar, no debería exceder el numero de hilos por bloque salvo que sincronicemos

#define N (50000)
#define HILOSPORBLOQUE (256)

__device__ float suma; //suma ubicada directamente en la GPU, se puede inicializar aqui con cudaMemcpyToSymbol
__device__ long long int ti[N]; //Variables globales en device para almacenar tiempos
__device__ long long int tf[N];

long long int h_ti[N]; //Variables de tiempos en el host 
long long int h_tf[N];

/**
 * CUDA Kernel Device code
 *  Suma de los elementos de un vector, versión que suma sobre una variable global suma en device
 */
__global__ void
sumVec(const float *A, int numElements) {
    
    cg::grid_group grid = cg::this_grid(); //creamos la estructura que refleja los hilos del grid
	long long int tini = clock64();

    int i = grid.thread_rank(); //Somos el hilo i dentro de todo el grid

    ti[i] = tini;

    if (i==0) { //El hilo 0 de todo el grid inicializa la suma
    	suma = 0.0;
    }

    //__syncthreads();
    grid.sync(); //Barrera para todos los hilos del grid

    if (i < numElements) {
    	atomicAdd(&suma,A[i]);
    }

    tf[i] = clock64();
}


/**
 * Host main routine
 */

int main(void) {

    // Vector length to be used, and compute its size
    const int numElements = N;
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vector

    for (int i = 0; i < numElements; i++) {
        h_A[i] = 20*rand()/(float)RAND_MAX; //inicializa a valores en [0.0, 20.0]
    	//h_A[i] = (float)i;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size));

    // Copy the host input vector A in host memory to the device input vector in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Obtiene un puntero usable desde kernel, aunque en este ejemplo no se usa. Atención al typecast!
    float* d_sum;
    CUDA_CHECK_RETURN(cudaGetSymbolAddress((void **)&d_sum, suma));

    int device = 0; // GPU 0
    cudaDeviceProp deviceProp;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, device));
    //Comprobar si funciona cooperative launch
    if (!deviceProp.cooperativeMultiDeviceLaunch) {
        printf("No se soporta 'cooperative kernel launch'\n");
        exit(EXIT_FAILURE);
    }
    printf("Hay %d SM's\n", deviceProp.multiProcessorCount);
    
    // Launch the sumVec CUDA Kernel
    //Bloques que nos harían falta
    int threadsPerBlock = HILOSPORBLOQUE;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    //Cuantos bloques cabrían en cada SM para el número de hilos y kernel concretos
    int numBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, sumVec, threadsPerBlock, 0);
    printf("Hay capacidad para ejecutar %d bloques por cada SM\n", numBlocksPerSm);
    
    //Preparamos el array de punteros a los parámetros
    void * args[] = { (void *) &d_A, (void *) &numElements };

    //Lanzamos el kernel si existe la capacidad suficiente para lanzar todos los bloques a la vez
    if (blocksPerGrid <= deviceProp.multiProcessorCount * numBlocksPerSm){
        printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
        CUDA_CHECK_RETURN(cudaLaunchCooperativeKernel((void*)sumVec, blocksPerGrid, threadsPerBlock, args));
    } else {
        printf("No hay recursos para lanzar simultáneamente %d bloques\n", blocksPerGrid);
        exit(EXIT_FAILURE);
    }
    //sumVec<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    //CUDA_CHECK_RETURN(cudaGetLastError());

    float sumatotal;
    // Copia el resultado en sumatotal
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&sumatotal, suma, sizeof(float)));
    // Copia los tiempos en las variables accesibles por el host
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_ti, ti, N*sizeof(long long int)));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_tf, tf, N*sizeof(long long int)));


    // Verify that the result is correct, en general se acumula mucho error en tipo float y en N grande puede ser mucho,
    // especialmente para valores sumados pequeños (pudiendo no ser conmutativa la suma)
    double sumcheck1 = 0.0;
    double sumcheck2 = 0.0;
    long long int t, tmin = LLONG_MAX, tmax = 0;
    int itmin, itmax;
    for (int i = 0; i < numElements; i++) {
    	t = h_tf[i] - h_ti[i];
    	if (h_tf[i] < h_ti[i]){
    		printf("Error en tiempos %u\n", i);
    	}
    	if (t < tmin){
    		tmin = t;
    		itmin = i;
    	}
    	if (t > tmax){
    	    tmax = t;
    	    itmax = i;
    	}
    	sumcheck1 = sumcheck1 + (double)h_A[i];
    	sumcheck2 = sumcheck2 + (double)h_A[numElements - i - 1]; //Otro orden de sumandos
    }

    #define TOL (1e-3)
    if ( (fabs(sumcheck1-sumatotal) > TOL) || (fabs(sumcheck2-sumatotal) > TOL) ){
        fprintf(stderr, "Result verification failed, da %f y los checks son %lf y %lf, errores de %lf y %lf\n", sumatotal, sumcheck1, sumcheck2, sumatotal - sumcheck1, sumatotal - sumcheck2);
        exit(EXIT_FAILURE);
    } else {
    	printf("Test PASSED\n");
    }
    printf("El tiempo máximo es %lld ciclos en hilo %d y el mínimo %lld ciclos en hilo %d\n",tmax, itmax, tmin, itmin);

    // Free device global memory
    CUDA_CHECK_RETURN(cudaFree(d_A));

    // Free host memory
    free(h_A);

    printf("Done\n");
    return EXIT_SUCCESS;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}

