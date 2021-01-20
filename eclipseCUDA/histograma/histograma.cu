#include <stdio.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define NUMELEMENTSV 33554432;
#define NUMELEMENTSH 8;
#define THREADSPERBLOCK 1024;
#define LOOPS 10;
/**
 * CUDA Kernel Device code
 * Calcula el histograma de un vector pasado. 
 * Versión directa con operaciones atómicas a vector H en memoria de video
 */
__global__ void histogram(int *V, int * H, int numElementsV, int numElementsH)
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
 * Versión directa con operaciones atómicas a memoria compartida por bloque y un solo acceso atómico final a la memoria global de la CPU, tras sincronización de los hilos
 * con el resultado a memorida de video.
 */
 __global__ void
 histogramShared(int *V, int * H, int numElementsV, int numElementsH)
 {
 __shared__ int acc[8];  ///tamaño máximo del vector H.Ver si se puede optimizar para  crear en tiempo de ejecución el vector dentro del kernel.
 ///además, no se como inicializar la variable a cero inicialmente.

  int i = blockDim.x * blockIdx.x + threadIdx.x;
   
  if (i % blockDim.x == 0)  ///si es el hilo del principio de un bloque, me encargaré de inicializar a cero la variable contador y después de ser el hilo que escriba a memoria global de la GPU.
    for (int j=0; j < 8; j++)
      acc[j] = 0;

     __syncthreads();    

 if (i < numElementsV)
   {
     int index = V[i] % numElementsH;
     atomicAdd((&acc[0] + index), 1); 
     
   }
   __syncthreads();    

   if (i % blockDim.x == 0)    
     for (int j = 0; j < numElementsH; j++)     
        atomicAdd((H + j), acc[j]); 
 
   
 }

/**
 * CUDA Kernel Device code
 * Suma por reducción de los elementos de un vector
 */
 __global__ void
 sumHistogram(int * h, int blocksPerGrid) ///blocksPerGrid dice el número de bloques  que realmente habría, o sea, los reales multiplicados por 2.
 {
     ///blockDim equivale al tamño del histograma, ya que se creó el bloque con tantos hilos como tamaño del histograma.

     int i = blockDim.x * blockIdx.x + threadIdx.x; ///posicion dentro del vector del histograma.
     int j = (blocksPerGrid -1 - blockIdx.x) * blockDim.x + threadIdx.x; ///valor  que voy a sumar dentro del histograma  que le corresponde. 
     ///O sea, el primer bloque(equivalente al tamaño  de elementos del histograma) se sumará con el último, el segundo, con el penúltimo.....

    h[i] = h[i] + h[j];
    
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
 * Versión directa con operaciones atómicas a memoria de video, pero con un histograma por bloque.
 * Se sigue utilizando un vector H, pero su tamaño es numElementsH * número de bloques.
 * O sea, hay un histograma de tamaño numElementsH por cada bloque y su ubicación en memoria es como vector, un histograma seguido del otro
 */
 __global__ void
 histogramByBlockShared(int *V, int * H, int numElementsV, int numElementsH)
 {
   __shared__ int acc[8];  ///tamaño máximo del vector H.Ver si se puede optimizar para  crear en tiempo de ejecución el vector dentro del kernel.
   ///además, no se como inicializar la variable a cero inicialmente.
  
    int i = blockDim.x * blockIdx.x + threadIdx.x;
     
    if (i % blockDim.x == 0)  ///si es el hilo del principio de un bloque, me encargaré de inicializar a cero la variable contador y después de ser el hilo que escriba a memoria global de la GPU.
      for (int j=0; j < 8; j++)
        acc[j] = 0;
  
       __syncthreads();    
  
     
     int index = V[i] % numElementsH;
    
     if (i < numElementsV)                  
        atomicAdd((&acc[0] + index), 1); 
            
     __syncthreads();    
  
     if (i % blockDim.x == 0)    
     {
        int index2 = blockIdx.x * numElementsH; ///me posiciono en histograma asociado a este bloque y en la posición correspondiente
        for (int j = 0; j < numElementsH; j++)     
        atomicAdd((H + index2 + j), acc[j]); 

     }


 }
/**
 * CUDA Kernel Device code
 * Calcula el histograma de un vector pasado mediante diferentes métodos.
 * Devuelve un puntero hacia el el vector del histograma ya calculado
 */
int * calculateHistogramByGpu(int * vector,int numElementsV, int numElementsH, bool byBlock, int threadsPerBlock, bool shared)
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
    //printf("\nCopy input data from the host memory vector V to the CUDA device");
    err = cudaMemcpy(d_V, vector, sizeV, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector V from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors H in host memory to the device input vectors in
    // device memory
    //printf("\nCopy input data from the host memory vector H to the CUDA device");
    err = cudaMemcpy(d_H, h_H, sizeH, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector H from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Launch the Vector Add CUDA Kernel   
	if (byBlock)	
	{
        if (shared)
        {

            printf("\nCUDA kernel - histogramByBlockShared - launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);       
            histogramByBlockShared<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_H, numElementsV, numElementsH);
            
            if ((blocksPerGrid % 2) !=  0) //si los bloques no son pares, 
                printf("\nBloques impares, todavía no implementada solución, puede fallar");       
    
            while (blocksPerGrid > 1)
            {
                
                sumHistogram<<<blocksPerGrid /2 , numElementsH>>>(d_H, blocksPerGrid);
                blocksPerGrid /= 2;
                
            }
    
        }
        else
        {

            printf("\nCUDA kernel - histogramByBlock - launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);       
            histogramByBlock<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_H, numElementsV, numElementsH);
            
            if ((blocksPerGrid % 2) !=  0) //si los bloques no son pares, 
                printf("\nBloques impares, todavía no implementada solución, puede fallar");       
    
            while (blocksPerGrid > 1)
            {
                
                sumHistogram<<<blocksPerGrid /2 , numElementsH>>>(d_H, blocksPerGrid);
                blocksPerGrid /= 2;
                
            }
    
        }
        
            

	}
	else
	{
        if (shared)
        {
            printf("\nCUDA kernel -histogramShared- launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);       
            histogramShared<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_H, numElementsV, numElementsH);        
        }
        else 
        {
            printf("\nCUDA kernel -histogram- launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);       
            histogram<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_H, numElementsV, numElementsH);        
    
        }
	
	}
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    // Copy the device result vector V in device memory to the host result vector
    // in host memory.
    //printf("\nCopy output data from the CUDA device vector V to the host memory");
    err = cudaMemcpy(vector, d_V, sizeV, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector V from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the device result vector H in device memory to the host result vector
    // in host memory.
    //printf("\nCopy output data from the CUDA device vector H to the host memory");
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

 */
int * calculateHistogramByCpu(int * vector, int numElementsV, int numElementsH)
{
////Version calculo histograma por CPU.     

int * h_H = (int *)malloc(numElementsH * sizeof(int));
  
// Verify that allocations succeeded
if (h_H == NULL)
{
    fprintf(stderr, "Failed to allocate host vector Histograma!\n");
    exit(EXIT_FAILURE);
}

unsigned t0,t1;

t0 = clock();

///Calculo el tiempo que tardaría si se hiciera por CPU
   for (int i = 0; i < numElementsV; i++)
      h_H[vector[i] % numElementsH] = h_H[vector[i] % numElementsH] +1;
     
 t1 = clock();
      double time =  (double (t1-t0)/CLOCKS_PER_SEC);
    
  
///Show Vector H
printf("\nTiempo empleado en calculo por CPU :  %f segundos",time);

return h_H ;
}

/**
 * Host main routine
 */
int main(void)
{

    int * h_H = NULL; ///puntero hacia el vector Histograma.
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElementsV = NUMELEMENTSV;
	int numElementsH = NUMELEMENTSH;
    int threadsPerBlock = THREADSPERBLOCK;
    int repeatLoop = LOOPS;
	    
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
   // printf("\nVector V Inicializado con :");

    for (int i = 0; i < numElementsV; ++i)    	 
		h_V[i] = rand() % 1000000;  ///número aleatorio entre 0 y 1000000
	 		//h_V[i] = i;

     
     

//*******************************************************************************************************************************   
////Version calculo histograma por CPU.     
//*******************************************************************************************************************************   

for (int r = 0; r < repeatLoop; r++)
{
     h_H = calculateHistogramByCpu(h_V, numElementsV, numElementsH);
    
    if (r == repeatLoop-1)  ///si es la última iteración, muestro los resultados
    {
         ///Show Vector H
        printf("\nResultado Vector Histograma Calculado por CPU ");
        for (int i = 0; i < numElementsH; ++i)    
            printf("\n[%d]", h_H[i]);

    } 
      
    free(h_H);    ///la función realiza la solicitud de memoria y debemos borrarla después de utilizarla antes de volver a llamar a la función
}
    
                    
    

    
    
//*******************************************************************************************************************************   
////Version calculo histograma por GPU. Versión, todos los hilos de todos los bloques a un mismo vector de histograma.
////Con acceso directo a memoria global de la GPU con escrituras atómicas.
//*******************************************************************************************************************************   
for (int r = 0; r < repeatLoop; r++)
{   
    h_H = calculateHistogramByGpu(h_V, numElementsV, numElementsH, false, threadsPerBlock,false);
    if (r == repeatLoop-1)  ///si es la última iteración, muestro los resultados
    {
         ///Show Vector H         
        printf("\nResultado Vector Histograma versión Sin bloque y Sin memoria compartida :");        
        for (int i = 0; i < numElementsH; ++i)    
            printf("\n[%d]", h_H[i]);

    } 
    free(h_H);
}


//*******************************************************************************************************************************   
////Version calculo histograma por GPU. Versión, todos los hilos de todos los bloques a un mismo vector de histograma.
////Con acceso directo a compartida por bloque mediante escrituras atómicas  y después solo el primer hilo de cada bloque escribirá, incrementando de forma
/// atómica los resultandos en la memoria global de la GPU. Al final son más escrituras atómicas, son las mismas que en la versión sin memoria compartida, más
/// una escritura, de todo el vector histograma, por bloque. Aún así, es tanta la diferencia de velocidad entre memoria global y compartida, que se mejora
///muchísimo el rendimiento.l
//*******************************************************************************************************************************   

for (int r = 0; r < repeatLoop; r++)
{   
    h_H = calculateHistogramByGpu(h_V, numElementsV, numElementsH, false, threadsPerBlock,true);

    if (r == repeatLoop-1)  ///si es la última iteración, muestro los resultados
    {
         ///Show Vector H                 
        printf("\nResultado Vector Histograma Versión Sin bloque y Memroria compartida :");
        for (int i = 0; i < numElementsH; ++i)    
            printf("\n[%d]", h_H[i]);

    } 

    free(h_H);
}


//*******************************************************************************************************************************   
////Version calculo histograma por GPU. Versión, todos los hilos de un mismo bloque, escribirán de forma atómica hacia
/// su propio vector histograma, de manera que las escrituras se reparten entre  los hilos de un histograma por bloque. Este vector histograma,
////realmente es uno por cada bloque, pero que se asignan como un único vector de tamaño "numero elementos vector histograma" * "numero de bloques a ejecutar"
/// Así que el acceso al histograma del primer bloque será la dirección del vector, la del segundo bloque, la dirección del vector + tamaño del numero de elemntos,(normalmente 8)
////Este algoritmo mejora, ya que parece hacer uso de los accesos a memoria mejorados de las GPU, ya sea por la optimización de escrituras/lecturas de palabras grandes
///como posibles paralelismos en las operaciones de lectura/escritura, también puede mejorar el hecho de que las operaciones atómicas, solo deben de protegerse de los hilos de un mismo bloque
/// y no de todos los bloques en ejecución.
///Este método requiere de la ejecución de dos kernel, el segundo es el encargado de sumar todos los histogramas generados por cada bloque y devolver 
// uno solo, que es colocado en las primeras posiciones del vector histograma general.
///La mejora es sustancial.
//*******************************************************************************************************************************   

for (int r = 0; r < repeatLoop; r++)
{   
    h_H = calculateHistogramByGpu(h_V, numElementsV, numElementsH, true, threadsPerBlock,false);   
    if (r == repeatLoop-1)  ///si es la última iteración, muestro los resultados
    {
         ///Show Vector H                 
         printf("\nResultado Vector Histograma Con bloques y Sin Memoria Compartida :");        
        for (int i = 0; i < numElementsH; ++i)    
            printf("\n[%d]", h_H[i]);

    } 
    free(h_H);
}



//*******************************************************************************************************************************   
////Version calculo histograma por GPU. Versión, todos los hilos de un mismo bloque, escribirán de forma atómica hacia
/// su propio vector histograma almacenado en memoria compartida para cada bloque. De manera que las escrituras se reparten entre  los hilos de un histograma por bloque.
/// Al final, solo el hilo cero de cada bloque, es el que se encarga de volcar su vector de memoria compartida (equivalente a las escrituras atómicas de todos los hilos de un bloque)
///  a la memoria global de la GPU. Este traspaso se realiza también mediante operaciones atómicas.
///Este método requiere de la ejecución de dos kernel, el segundo es el encargado de sumar todos los histogramas generados por cada bloque y devolver 
// uno solo, que es colocado en las primeras posiciones del vector histograma general.

//*******************************************************************************************************************************   

for (int r = 0; r < repeatLoop; r++)
{   
    h_H = calculateHistogramByGpu(h_V, numElementsV, numElementsH, true, threadsPerBlock,true);   
    if (r == repeatLoop-1)  ///si es la última iteración, muestro los resultados
    {
         ///Show Vector H                          
         printf("\nResultado Vector Histograma Con bloques y Memoria Compartida :");
        for (int i = 0; i < numElementsH; ++i)    
            printf("\n[%d]", h_H[i]);

    } 
    free(h_H);

}

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
