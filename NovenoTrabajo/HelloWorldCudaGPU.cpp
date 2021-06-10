#include <stdio.h>

__global__ void hello() {
    printf("Hello world from device");
}

int main() {
    hello<<<1, 1>>>();
    printf("Hello world from host");
    cudaDeviceSynchronize()；
    return 0;
}