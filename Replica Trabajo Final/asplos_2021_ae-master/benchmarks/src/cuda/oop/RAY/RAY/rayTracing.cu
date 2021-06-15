/*
 * Copyright 2008 BOROUJERDI Maxime. Tous droits reserves.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "makebmp.h"

#include <cutil.h>
#include <helper_timer.h>
#include <rayTracing_kernel.cu>

#define PI 3.141592654f
#define Angle(a) ((a * PI) / 180.0)

int g_verbose;
int t = 1;

class Observateur {
  private:
    matrice3x4 M;  // U, V, W
    float df;      // distance focale

  public:
    Observateur();
    Observateur(const float3&, const float3&, const float3&, double);

    inline const matrice3x4& getMatrice() const { return M; }
    inline float getDistance() const { return df; }
};

Observateur::Observateur() {
    M.m[0] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    M.m[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    M.m[2] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    df = 1.0 / tan(Angle(65) / 2.0);
}

Observateur::Observateur(const float3& p, const float3& u, const float3& v,
                         double a) {
    float3 VP, U, V, W;
    VP = normalize(v);
    U = normalize(u);
    V = normalize(VP - dot(U, VP) * U);
    W = normalize(cross(U, V));
    M.m[0] = make_float4(U.x, U.y, U.z, p.x);
    M.m[1] = make_float4(V.x, V.y, V.z, p.y);
    M.m[2] = make_float4(W.x, W.y, W.z, p.z);
    df = 1.0 / tan(Angle(a) / 2.0);
}
Observateur obs = Observateur(
    make_float3(0.0f, 0.5f, 2.0f),
    normalize(make_float3(0.0f, 0.0f, 0.0f) - make_float3(0.0f, 0.5f, 2.0f)),
    make_float3(0.0f, 1.0f, 0.0f), 65.0f);

#include <rayTracing_kernel.cu>

unsigned width = 64;   // 640; //512; //16; //32; //512;
unsigned height = 64;  // 480; //512; //16;//512;
dim3 blockSize(16, 8);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

StopWatchInterface* timer = NULL;

uint *c_output, *d_output;

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void initPixelBuffer() {
    // int num = width * height;
    // float phi = 2.0f/(float)min(width,height);
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
};

// Rendu de l'image avec CUDA
void render(Object** objList, int n) {
    sdkStartTimer(&timer);
    render<<<gridSize, blockSize>>>(d_output, objList, width, height,
                                    obs.getDistance(), n);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    CUDA_SAFE_CALL(cudaMemcpy(c_output, d_output, width * height * sizeof(uint),
                              cudaMemcpyDeviceToHost));
    unsigned long long int checksum = 0;
    for (int y = (height - 1); y >= 0; y--) {
        if (g_verbose) printf("\n");
        for (int x = 0; x < width; x++) {
            if (g_verbose) printf("%010u ", (unsigned)c_output[x + y * width]);
            checksum += c_output[x + y * width];
        }
    }
    printf("\n");
    printf("checksum=%llx\n", checksum);
}

// Affichage du resultat avec OpenGL
void display(Object** objList, int n) {
    // Affichage du resultat
    render(objList, n);
    printf("Kernel Time: %f \n", sdkGetTimerValue(&timer));

    t--;
    if (!t) {
        return;
    }
}
////////////////////////////////////////////////////////////////////////////////
// Programme principal
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    // initialise card and timer
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "There is no device.\n");
        exit(EXIT_FAILURE);
    }
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
        if (deviceProp.major >= 1) break;
    }
    if (dev == deviceCount) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        exit(EXIT_FAILURE);
    } else
        CUDA_SAFE_CALL(cudaSetDevice(dev));
    int i, commandline_error;
    commandline_error = 0;
    g_verbose = 0;
    if (argc >= 4) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        for (i = 4; i < argc; i++) {
            if (argv[i][0] == '-') {
                switch (argv[i][1]) {
                    case 'v':
                        g_verbose = 1;
                        break;
                    default:
                        commandline_error = 1;
                }
            } else
                commandline_error = 1;
        }
    } else
        commandline_error = 1;

    if (commandline_error || !width || !height) {
        printf("Usage: ./rayTracing <WIDTH> <HEIGHT> [-v]\n");
        printf(
            "where WIDTH and HEIGHT are the screen dimensions and -v is used "
            "to display an abstract representation of the output.\n");
        return 1;
    }
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    initialize_bmp(width, height, 32);

    Object** objList;
    int n = atoi(argv[3]);
    float* A;
    float* d_A;
    A = (float*)malloc(n * 8 * sizeof(float));
    cudaMalloc(&d_A, n * 8 * sizeof(float));
    srand(47);
    A[0] = 0.0f;
    A[1] = 1.0f;
    A[2] = 1.0f;
    A[3] = 1.0f;
    A[4] = 0.0f;
    A[5] = -1.5f;
    A[6] = -0.0f;
    A[7] = 0.5f;
    A[8] = 1.0f;
    A[8 + 1] = 0.0f;
    A[8 + 2] = 0.0f;
    A[8 + 3] = 1.0f;
    A[8 + 4] = -1.0f;
    A[8 + 5] = 0.0f;
    A[8 + 6] = -1.0f;
    A[8 + 7] = 0.5f;
    A[16] = 0.0f;
    A[16 + 1] = 0.0f;
    A[16 + 2] = 1.0f;
    A[16 + 3] = 1.0f;
    A[16 + 4] = 1.0f;
    A[16 + 5] = -0.0f;
    A[16 + 6] = -1.0f;
    A[16 + 7] = 0.5f;
    A[24] = 0.0f;
    A[24 + 1] = 1.0f;
    A[24 + 2] = 0.0f;
    A[24 + 3] = 1.0f;
    A[24 + 4] = 0.0f;
    A[24 + 5] = -0.0f;
    A[24 + 6] = -2.0f;
    A[24 + 7] = 0.75f;
    for (int i(4); i < n; i++) {
        float r, v, b;
        float tmp1(5.0f * ((r = (float(rand() % 255) / 255.0f))) - 2.5f);
        float tmp2(5.0f * ((v = (float(rand() % 255) / 255.0f))) - 2.5f);
        float tmp3(-5.0f * ((b = (float(rand() % 255) / 255.0f))));
        float tmp4((rand() % 100) / 100.0f);
        A[i * 8 + 4] = tmp1;
        A[i * 8 + 5] = tmp2;
        A[i * 8 + 6] = tmp3;
        A[i * 8 + 7] = tmp4;
        A[i * 8] = r;
        A[i * 8 + 1] = v;
        A[i * 8 + 2] = b;
        A[i * 8 + 3] = 1.0f;
    }
    cudaMemcpy(d_A, A, n * 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&objList, sizeof(Object*) * n);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    initObject<<<blocksPerGrid, threadsPerBlock>>>(objList, d_A, n);
    cudaDeviceSynchronize();
    c_output = (uint*)calloc(width * height, sizeof(uint));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&d_output, width * height * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MView, (void*)&obs, 3 * sizeof(float4)));
    initPixelBuffer();
    display(objList, n);
    create_bmp(c_output);
    sdkDeleteTimer(&timer);
    return 0;
}
