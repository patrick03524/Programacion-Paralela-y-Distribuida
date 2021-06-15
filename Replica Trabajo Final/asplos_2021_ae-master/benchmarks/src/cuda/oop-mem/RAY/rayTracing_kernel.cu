/*
 * Copyright 2008 BOROUJERDI Maxime. Tous droits reserves.
 */

#ifndef __RAYTRACING_KERNEL_H__
#define __RAYTRACING_KERNEL_H__

#include "../../mem_alloc/mem_alloc.h"
#include "helper_math.h"

#define numObj 5

class inputs {
  public:
    inputs() {
        this->R = 0.0f;
        this->V = 0.0f;
        this->B = 0.0f;
        this->A = 0.0f;
        this->R = 0.0f;
        this->C = make_float3(0.0f, 0.0f, 0.0f);
    }
    int is_sphere;
    float R, V, B, A;
    float3 C;  // centre
    float r;   // rayon
};

typedef struct {
    float4 m[3];
} matrice3x4;

__constant__ matrice3x4 MView;  // matrice inverse de la matrice de vue

__device__ float float2int_pow50(float a) {
    return a * a * a * a * a * a * a * a * a * a * a * a * a * a * a * a * a *
           a * a * a * a * a * a * a * a * a * a * a * a * a * a * a * a * a *
           a * a * a * a * a * a * a * a * a * a * a * a * a * a * a * a;
}

__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp entre [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
           (uint(rgba.y * 255) << 8) | (uint(rgba.x * 255));
}

class Rayon {
  public:
    float3 A;  // origine
    float3 u;  // direction
};

class Object {
  public:
    __device__ Object() {
        this->R = 0.0f;
        this->V = 0.0f;
        this->B = 0.0f;
        this->A = 0.0f;
        this->R = 0.0f;
        this->C = make_float3(0.0f, 0.0f, 0.0f);
    }
    __device__ Object(float R, float V, float B, float A, float3 C, float r) {
        this->R = R;
        this->V = V;
        this->B = B;
        this->A = A;
        this->C = C;
        this->r = r;
    }
    __noinline__ __device__ virtual float intersection(Rayon R) = 0;
    __noinline__ __device__ virtual float3 getNormale(float3 P) = 0;
    float R, V, B, A;
    float3 C;  // centre
    float r;   // rayon
};

class Plain : public Object {
  public:
    __device__ Plain() {}
    __device__ Plain(float R, float V, float B, float A, float3 C, float r) {
        this->R = R;
        this->V = V;
        this->B = B;
        this->A = A;
        this->C = C;
        this->r = r;
    }
    __noinline__ __device__ float3 getNormale(float3 P) {
        return normalize(make_float3(0.0f, 1.0f, 0.0f));
    }
    __noinline__ __device__ float intersection(Rayon R) {
        float res;
        float3 N = normalize(make_float3(0.0f, 1.0f, 0.0f));
        float m(dot(N, R.u)), d, t;
        float3 L;

        if (fabs(m) < 0.0001f) {
            res = 0.0f;
        } else {
            L = R.A - this->C;
            d = dot(N, L);
            t = -d / m;
            if (t > 0) {
                res = t;
            } else {
                res = 0.0f;
            }
        }

        return res;
    }
};

class Sphere : public Object {
  public:
    __device__ Sphere() {}
    __device__ Sphere(float R, float V, float B, float A, float3 C, float r) {
        this->R = R;
        this->V = V;
        this->B = B;
        this->A = A;
        this->C = C;
        this->r = r;
    }
    __noinline__ __device__ float3 getNormale(float3 P) {
        return normalize(P - this->C);
    }
    __noinline__ __device__ float intersection(Rayon R) {
        float3 L(this->C - R.A);
        float d(dot(L, R.u)), l2(dot(L, L)), r2(this->r * this->r), m2, q, res;

        if (d < 0.0f && l2 > r2) {
            res = 0.0f;
        } else {
            m2 = l2 - d * d;
            if (m2 > r2) {
                res = 0.0f;
            } else {
                q = sqrt(r2 - m2);
                if (l2 > r2)
                    res = d - q;
                else
                    res = d + q;
            }
        }

        return res;
    }
};

void initObject(Object **objList, float *A, int n, obj_alloc *alloc) {
    int i = 0;
    for (i = 0; i < n; i++) {
        if (i == 0) {
            objList[i] = (Object *)alloc->my_new<Plain>();
        } else {
            objList[i] = (Object *)alloc->my_new<Sphere>();
        }
    }
}
__global__ void initObject_kern(Object **objList, float *A, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        if (i == 0) {
            new (objList[i])
                Plain(A[i * 8], A[i * 8 + 1], A[i * 8 + 2], A[i * 8 + 3],
                      make_float3(A[i * 8 + 4], A[i * 8 + 5], A[i * 8 + 6]),
                      A[i * 8 + 7]);
        } else {
            new (objList[i])
                Sphere(A[i * 8], A[i * 8 + 1], A[i * 8 + 2], A[i * 8 + 3],
                       make_float3(A[i * 8 + 4], A[i * 8 + 5], A[i * 8 + 6]),
                       A[i * 8 + 7]);
        }
    }
}
__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size_g;

__managed__ void *temp_render;
__device__ bool notShadowRay(Object **__restrict__ objList, float3 A, float3 u,
                             int NUM) {
    float t(0.0f);
    Rayon ray;

    float3 L(make_float3(10.0f, 10.0f, 10.0f)), tmp;
    float dst(dot(tmp = (L - A), tmp));
    ray.A = A + u * 0.0001f;
    ray.u = u;
    for (int j = 0; j < NUM && !t; j++) {
        t = objList[j]->intersection(ray);
        if (t > 0.0f && dot(tmp = (A + u * t), tmp) > dst) {
            t = 0.0f;
        }
    }
    return t == 0.0f;
}

__device__ bool notShadowRay_vptr(Object **__restrict__ objList, float3 A,
                                  float3 u, int NUM, range_tree_node *table) {
    float t(0.0f);
    Rayon ray;
    unsigned tree_size = tree_size_g;

    void **vtable;

    float3 L(make_float3(10.0f, 10.0f, 10.0f)), tmp;
    float dst(dot(tmp = (L - A), tmp));
    ray.A = A + u * 0.0001f;
    ray.u = u;
    for (int j = 0; j < NUM && !t; j++) {
        vtable = get_vfunc(objList[j], table, tree_size);
        temp_render = vtable[0];
        t = objList[j]->intersection(ray);
        if (t > 0.0f && dot(tmp = (A + u * t), tmp) > dst) {
            t = 0.0f;
        }
    }
    return t == 0.0f;
}

__global__ void render_vptr(uint *result, Object **__restrict__ objList,
                            uint imageW, uint imageH, float df, int NUM) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint tid(__umul24(threadIdx.y, blockDim.x) + threadIdx.x);
    unsigned tree_size = tree_size_g;
    // range_tree_node *table = range_tree;
    void **vtable;
    __shared__ range_tree_node table[3];
    if (threadIdx.x < tree_size && threadIdx.y == 0) {
        // for (int i = 0; i < tree_size; i++) {
        // printf("%d\n",threadIdx.x);
        memcpy(&table[threadIdx.x], &range_tree[threadIdx.x],
               sizeof(range_tree_node));
        // if(tid==0)
        // printf("%p %p \n",table[i].range_start,table[i].range_end);
    }

    uint id(x + y * imageW);
    float4 pile[5];
    uint Obj, n = 0, nRec = 5;
    float prof, tmp;

    for (int i = 0; i < nRec; ++i)
        pile[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    if (x < imageW && y < imageH) {
        prof = 10000.0f;
        result[id] = 0;
        float tPixel(2.0f / float(min(imageW, imageH)));
        float4 f(make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        matrice3x4 M(MView);
        Rayon R;
        R.A = make_float3(M.m[0].w, M.m[1].w, M.m[2].w);
        R.u = make_float3(M.m[0]) * df +
              make_float3(M.m[2]) * (float(x) - float(imageW) * 0.5f) * tPixel +
              make_float3(M.m[1]) * (float(y) - float(imageH) * 0.5f) * tPixel;
        R.u = normalize(R.u);
        __syncthreads();

        // printf("%d: nRec %d\n", threadIdx.x, nRec);
        for (int i = 0; i < nRec && n == i; i++) {
            for (int j = 0; j < NUM; j++) {
                float t;
                // vtable = get_vfunc(objList[j], table, tree_size);
                // temp_render = vtable[0];
                t = objList[j]->intersection(R);
                if (t > 0.0f && t < prof) {
                    prof = t;
                    Obj = j;
                }
            }
            // printf("%d: i=%d, t=%e\n", threadIdx.x, i, prof);
            float t = prof;
            if (t > 0.0f && t < 10000.0f) {
                n++;
                float4 color(make_float4(objList[Obj]->R, objList[Obj]->V,
                                         objList[Obj]->B, objList[Obj]->A));
                float3 P(R.A + R.u * t),
                    L(normalize(make_float3(10.0f, 10.0f, 10.0f) - P)),
                    V(normalize(R.A - P));
                vtable = get_vfunc(objList[Obj], table, tree_size);
                temp_render = vtable[1];
                float3 N(objList[Obj]->getNormale(P));
                float3 Np(dot(V, N) < 0.0f ? (-1 * N) : N);
                pile[i] = 0.05f * color;
                if (dot(Np, L) > 0.0f &&
                    notShadowRay_vptr(objList, P, L, NUM, table)) {
                    // float3 Ri(2.0f*Np*dot(Np,L) - L);
                    float3 Ri(normalize(L + V));
                    // Ri = (L+V)/normalize(L+V);
                    pile[i] += 0.3f * color * (min(1.0f, dot(Np, L)));
                    tmp = 0.8f * pow(max(0.0f, min(1.0f, dot(Np, Ri))), 50.0f);
                    // tmp = 0.8f *
                    // float2int_pow50(max(0.0f,min(1.0f,dot(Np,Ri))));
                    pile[i].x += tmp;
                    pile[i].y += tmp;
                    pile[i].z += tmp;
                }

                R.u = 2.0f * N * dot(N, V) - V;
                R.u = normalize(R.u);
                R.A = P + R.u * 0.0001f;
            }
            prof = 10000.0f;
        }
        for (int i(n - 1); i > 0; i--)
            pile[i - 1] = pile[i - 1] + 0.8f * pile[i];
        result[id] += rgbaFloatToInt(pile[0]);
    }
}
__global__ void render(uint *result, Object **__restrict__ objList, uint imageW,
                       uint imageH, float df, int NUM) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint tid(__umul24(threadIdx.y, blockDim.x) + threadIdx.x);

    uint id(x + y * imageW);
    float4 pile[5];
    uint Obj, n = 0, nRec = 5;
    float prof, tmp;

    for (int i = 0; i < nRec; ++i)
        pile[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    if (x < imageW && y < imageH) {
        prof = 10000.0f;
        result[id] = 0;
        float tPixel(2.0f / float(min(imageW, imageH)));
        float4 f(make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        matrice3x4 M(MView);
        Rayon R;
        R.A = make_float3(M.m[0].w, M.m[1].w, M.m[2].w);
        R.u = make_float3(M.m[0]) * df +
              make_float3(M.m[2]) * (float(x) - float(imageW) * 0.5f) * tPixel +
              make_float3(M.m[1]) * (float(y) - float(imageH) * 0.5f) * tPixel;
        R.u = normalize(R.u);
        __syncthreads();

        // printf("%d: nRec %d\n", threadIdx.x, nRec);
        for (int i = 0; i < nRec && n == i; i++) {
            for (int j = 0; j < NUM; j++) {
                float t;
                t = objList[j]->intersection(R);
                if (t > 0.0f && t < prof) {
                    prof = t;
                    Obj = j;
                }
            }
            // printf("%d: i=%d, t=%e\n", threadIdx.x, i, prof);
            float t = prof;
            if (t > 0.0f && t < 10000.0f) {
                n++;
                float4 color(make_float4(objList[Obj]->R, objList[Obj]->V,
                                         objList[Obj]->B, objList[Obj]->A));
                float3 P(R.A + R.u * t),
                    L(normalize(make_float3(10.0f, 10.0f, 10.0f) - P)),
                    V(normalize(R.A - P));
                float3 N(objList[Obj]->getNormale(P));
                float3 Np(dot(V, N) < 0.0f ? (-1 * N) : N);
                pile[i] = 0.05f * color;
                if (dot(Np, L) > 0.0f && notShadowRay(objList, P, L, NUM)) {
                    // float3 Ri(2.0f*Np*dot(Np,L) - L);
                    float3 Ri(normalize(L + V));
                    // Ri = (L+V)/normalize(L+V);
                    pile[i] += 0.3f * color * (min(1.0f, dot(Np, L)));
                    tmp = 0.8f * pow(max(0.0f, min(1.0f, dot(Np, Ri))), 50.0f);
                    // tmp = 0.8f *
                    // float2int_pow50(max(0.0f,min(1.0f,dot(Np,Ri))));
                    pile[i].x += tmp;
                    pile[i].y += tmp;
                    pile[i].z += tmp;
                }

                R.u = 2.0f * N * dot(N, V) - V;
                R.u = normalize(R.u);
                R.A = P + R.u * 0.0001f;
            }
            prof = 10000.0f;
        }
        for (int i(n - 1); i > 0; i--)
            pile[i - 1] = pile[i - 1] + 0.8f * pile[i];
        result[id] += rgbaFloatToInt(pile[0]);
    }
}
#endif  // __RAYTRACING_KERNEL_H__
