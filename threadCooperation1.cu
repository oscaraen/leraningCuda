
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

using namespace std;

const int threadsPerBlock = 4;
// kernel que hace la suma 
__global__ void sumarVectores(float* vector1, float* vector2, int tam) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x; // pendiente saber como cambio blockDIm
    while (tid < tam) {
        printf("threadIdx: %d, blockIdx:%d, blockDim:%d, gridDim:%d tid: %d\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, tid);
        vector2[tid] += vector1[tid];
        tid += blockDim.x * gridDim.x; // el truco para hacer vectores muy largos o shared memory es sumarle blockDim+gridDim
    }
}


// Kernel para hacer producto punto con shared memory
__global__ void productoPunto(float* vector1, float* vector2, int tam, float * output) {
    __shared__ float cache[threadsPerBlock]; // creo una caché con thre
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    // parte 1: 
    while (tid < tam) {
        temp += vector1[tid] * vector2[tid];
        printf("tid: %d, temp is: %.2f\n", tid, temp); //primera ejecución, para vector arbitrario hay saltos de blockDim*gridDim
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;
    // wait for all threads end
    __syncthreads();
    // para las reducciones el número de threads por bloque debe ser potencia de 2 para poder hacerla efectiva sin que sobren ni falten elementos
    int i = blockDim.x / 2; // blockDim dice cuantos threads per block hay
    while (i != 0) {
        float testVariable = 0;
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        testVariable = cache[cacheIndex];
        printf("cacheindex: %.2f\n", testVariable);
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        output[blockIdx.x] = cache[0]; // al final en cada 16th voy a tener la suma de todos los demás, hay que sumar esos


}

/** 
* Thread cooperation example:
* 
* Siguiendo el libro cuda_by_example a ver si es posible usar shared memory para alguna operación
*/
int main()
{
    // En cuda tenemos hiulos que hacen copias en paralelo de un código (kernel) (bloques)
    // Cuda permite que estos bloques se separen en threads
    // en el llamado se manda <<<bloques, hilos>>>

    float *vector1;
    float* vector2;

    vector1 = (float*)malloc(sizeof(float) * 500);
    vector2 = (float*)malloc(sizeof(float) * 500);
    for (int i = 0; i < 500; i++) {
        // Llenar los vectores con números cualquiera
        vector1[i] =1;
        vector2[i] = 1;
    }
    cout << vector1[0] << " v2: " << vector2[0] << endl;

    // reservar memoria en cuda
    float* d_vector1, * d_vector2, *d_vector3;
    cudaMalloc(&d_vector1, sizeof(float) * 500);
    cudaMalloc(&d_vector2, sizeof(float) * 500);
    cudaMalloc(&d_vector3, sizeof(float) * 500);
    // copiar datos a cuda
    cudaMemcpy(d_vector1, vector1, sizeof(float) * 500, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, vector2, sizeof(float) * 500, cudaMemcpyHostToDevice);

    productoPunto <<<128, 128>>> (d_vector1, d_vector2, 100, d_vector3);

    cudaMemcpy(vector1, d_vector3, sizeof(float) * 500, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; i++) {
        cout << vector1[i] << ", ";
    }
    cout << endl;


    cout << "Si lees esto, todo salio bien" << endl;

    
    return 0;
}

