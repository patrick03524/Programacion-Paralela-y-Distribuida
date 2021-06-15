/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include "../../mem_alloc/mem_alloc.h"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

class BaseClass {
  public:
    virtual __device__ void doTheMath(float &c, float a, int numCompute) = 0;
};

#define Derived(A)                                                             \
    class Class##A : public BaseClass {                                        \
      public:                                                                  \
        virtual __device__ void doTheMath(float &c, float a, int numCompute) { \
            for (int l = 0; l < numCompute; l++) c = c + a;                    \
        }                                                                      \
    }

Derived(0);
Derived(1);
Derived(2);
Derived(3);
Derived(4);
Derived(5);
Derived(6);
Derived(7);
Derived(8);
Derived(9);
Derived(10);
Derived(11);
Derived(12);
Derived(13);
Derived(14);
Derived(15);
Derived(16);
Derived(17);
Derived(18);
Derived(19);
Derived(20);
Derived(21);
Derived(22);
Derived(23);
Derived(24);
Derived(25);
Derived(26);
Derived(27);
Derived(28);
Derived(29);
Derived(30);
Derived(31);

#define ObjCase_cpu(A)                                         \
    case A:                                                    \
        if (numElements > i) {                                 \
            array[i] = (BaseClass *)alloc->my_new<Class##A>(); \
            break;                                             \
        }

#define ObjCase(A)                     \
    case A:                            \
        if (numElements > i) {         \
            new (array[i]) Class##A(); \
            break;                     \
        }

__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size_g;

__managed__ void *temp_ubench;
void initialize_0(BaseClass **pointerArray, int numElements, int numClasses,
                  int threadsPerBlock, obj_alloc *alloc) {
    int i;
    int threadIdx;
    BaseClass **array = pointerArray;
    for (i = 0; i < numElements; i++) {
        threadIdx = i ;/// threadsPerBlock;

        switch (threadIdx % numClasses) {
            ObjCase_cpu(0);
            ObjCase_cpu(1);
            ObjCase_cpu(2);
            ObjCase_cpu(3);
            ObjCase_cpu(4);
            ObjCase_cpu(5);
            ObjCase_cpu(6);
            ObjCase_cpu(7);
            ObjCase_cpu(8);
            ObjCase_cpu(9);
            ObjCase_cpu(10);
            ObjCase_cpu(11);
            ObjCase_cpu(12);
            ObjCase_cpu(13);
            ObjCase_cpu(14);
            ObjCase_cpu(15);
            ObjCase_cpu(16);
            ObjCase_cpu(17);
            ObjCase_cpu(18);
            ObjCase_cpu(19);
            ObjCase_cpu(20);
            ObjCase_cpu(21);
            ObjCase_cpu(22);
            ObjCase_cpu(23);
            ObjCase_cpu(24);
            ObjCase_cpu(25);
            ObjCase_cpu(26);
            ObjCase_cpu(27);
            ObjCase_cpu(28);
            ObjCase_cpu(29);
            ObjCase_cpu(30);
            ObjCase_cpu(31);
        }
    }
}
__global__ void initialize_1(BaseClass **pointerArray, int numElements,
                             int numClasses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    BaseClass **array = pointerArray;
    switch (threadIdx.x % numClasses) {
        ObjCase(0);
        ObjCase(1);
        ObjCase(2);
        ObjCase(3);
        ObjCase(4);
        ObjCase(5);
        ObjCase(6);
        ObjCase(7);
        ObjCase(8);
        ObjCase(9);
        ObjCase(10);
        ObjCase(11);
        ObjCase(12);
        ObjCase(13);
        ObjCase(14);
        ObjCase(15);
        ObjCase(16);
        ObjCase(17);
        ObjCase(18);
        ObjCase(19);
        ObjCase(20);
        ObjCase(21);
        ObjCase(22);
        ObjCase(23);
        ObjCase(24);
        ObjCase(25);
        ObjCase(26);
        ObjCase(27);
        ObjCase(28);
        ObjCase(29);
        ObjCase(30);
        ObjCase(31);
    }
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void ooVectorAdd(const float *A, float *C, int numElements,
                            BaseClass **classes, int numCompute) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    BaseClass *myClass = classes[i];
    unsigned tree_size = tree_size_g;
    range_tree_node *table = range_tree;
    void **vtable;
    if (i < numElements) {
        vtable = get_vfunc(myClass, table, tree_size);
        temp_ubench = vtable[0];
        myClass->doTheMath(C[i], A[i], numCompute);
    }
}

/**
 * Host main routine
 */
int main(int argc, char **argv) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
    obj_alloc my_obj_alloc(&shared_mem);
    // Print the vector length to be used, and compute its size
    int numElements = atoi(argv[1]);  // size of vector
    int numCompute = atoi(argv[3]);   // vfunc body size
    int numClasses = atoi(argv[4]);   // num of types
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);
    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    BaseClass **classes = NULL;
    // cudaMalloc((void***)&classes, sizeof(BaseClass*)*numElements);
    classes = (BaseClass **)my_obj_alloc.calloc<BaseClass *>(numElements);
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = atoi(argv[2]);  // thread per block
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    initialize_0(classes, numElements, numClasses, threadsPerBlock,
                 &my_obj_alloc);
    initialize_1<<<blocksPerGrid, threadsPerBlock>>>(classes, numElements,
                                                     numClasses);
                                                     cudaDeviceSynchronize();
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch initialize kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    my_obj_alloc.create_tree();
    range_tree = my_obj_alloc.get_range_tree();
    tree_size_g = my_obj_alloc.get_tree_size();
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);
    ooVectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements,
                                                    classes, numCompute);
    cudaDeviceSynchronize();
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to launch ooVectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        float result = 0;
        for (int j = 0; j < numCompute; j++) result += h_A[i];
        if (fabs(result - h_C[i]) > 1e-3) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
