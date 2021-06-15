#include "parse_oo.h"

void initContext(GraphChiContext *context, int vertices, int edges) {
    // int tid = blockDim.x * blockIdx.x + threadIdx.x;

    context->setNumIterations(0);
    context->setNumVertices(vertices);
    context->setNumEdges(edges);
}



void part0_initObject(ChiVertex<int, int> **vertex, GraphChiContext *context,
                      int *row, int *col, int *inrow, int *incol,
                      obj_alloc *alloc) {
    int tid = 0;

    for (tid = 0; tid < context->getNumVertices(); tid++) {
        vertex[tid] =
            (ChiVertex<int, int> *)alloc->calloc<ChiVertex<int, int>>(1);
    }
}
void part1_initObject(ChiVertex<int, int> **vertex, GraphChiContext *context,
                      int *row, int *col, int *inrow, int *incol,
                      obj_alloc *alloc) {
    int tid = 0;

    for (tid = 0; tid < context->getNumVertices(); tid++) {
        // int out_start = row[tid];
        // int out_end;
        // if (tid + 1 < context->getNumVertices()) {
        //   out_end = row[tid + 1];
        // } else {
        //   out_end = context->getNumEdges();
        // }
        // int in_start = inrow[tid];
        // int in_end;
        // if (tid + 1 < context->getNumVertices()) {
        //   in_end = inrow[tid + 1];
        // } else {
        //   in_end = context->getNumEdges();
        // }
        // int indegree = in_end - in_start;
        // int outdegree = out_end - out_start;
        // vertex[tid].inEdgeDataArray =
        //     (ChiEdge<myType> *)alloc->my_new<Edge<myType>>(indegree);
        // vertex[tid].outEdgeDataArray =
        //     (ChiEdge<myType> **)alloc->my_new<Edge<myType> *>(outdegree);
        // new (&vertex[tid]) ChiVertex<int, int>(tid, indegree,
        // outdegree,alloc);
        vertex[tid]->set_in_out(alloc);
        // vertex[tid].setValue(INT_MAX);
        // for (int i = in_start; i < in_end; i++) {
        //   vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
        // }
        // for (int i = out_start; i < out_end; i++) {
        //    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        //}
    }
}
__global__ void part_kern0_initObject(ChiVertex<int, int> **vertex,
                                      GraphChiContext *context, int *row,
                                      int *col, int *inrow, int *incol) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < context->getNumVertices()) {
        int out_start = row[tid];
        int out_end;
        if (tid + 1 < context->getNumVertices()) {
            out_end = row[tid + 1];
        } else {
            out_end = context->getNumEdges();
        }

        int in_start = inrow[tid];
        int in_end;
        if (tid + 1 < context->getNumVertices()) {
            in_end = inrow[tid + 1];
        } else {
            in_end = context->getNumEdges();
        }
        int indegree = in_end - in_start;
        int outdegree = out_end - out_start;

        new (vertex[tid]) ChiVertex<int, int>(tid, indegree, outdegree);

        // vertex[tid].setValue(INT_MAX);
        // for (int i = in_start; i < in_end; i++) {
        //   vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
        // }
    }
}
__global__ void part_kern1_initObject(ChiVertex<int, int> **vertex,
                                      GraphChiContext *context, int *row,
                                      int *col, int *inrow, int *incol) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < context->getNumVertices()) {
        // int out_start = row[tid];
        // int out_end;
        // if (tid + 1 < context->getNumVertices()) {
        //   out_end = row[tid + 1];
        // } else {
        //   out_end = context->getNumEdges();
        // }

        int in_start = inrow[tid];
        int in_end;
        if (tid + 1 < context->getNumVertices()) {
            in_end = inrow[tid + 1];
        } else {
            in_end = context->getNumEdges();
        }

        vertex[tid]->setValue(INT_MAX);
        for (int i = in_start; i < in_end; i++) {
            vertex[tid]->setInEdge(i - in_start, incol[i], INT_MAX);
        }
    }
}
void initOutEdge(ChiVertex<int, int> **vertex, GraphChiContext *context,
                 int *row, int *col) {
    int tid = 0;

    for (tid = 0; tid < context->getNumVertices(); tid++) {
        int out_start = row[tid];
        int out_end;
        if (tid + 1 < context->getNumVertices()) {
            out_end = row[tid + 1];
        } else {
            out_end = context->getNumEdges();
        }

        for (int i = out_start; i < out_end; i++) {
            vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i],
                                    INT_MAX);
        }
    }
}


__global__ void kern_initOutEdge(ChiVertex<int, int> **vertex,
                                 GraphChiContext *context, int *row, int *col) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        int out_start = row[tid];
        int out_end;
        if (tid + 1 < context->getNumVertices()) {
            out_end = row[tid + 1];
        } else {
            out_end = context->getNumEdges();
        }
        // int in_start = inrow[tid];
        // int in_end;
        // if (tid + 1 < context->getNumVertices()) {
        //    in_end = inrow[tid + 1];
        //} else {
        //    in_end = context->getNumEdges();
        //}
        // int indegree = in_end - in_start;
        // int outdegree = out_end - out_start;
        // vertex[tid] = new ChiVertex<float, float>(tid, indegree, outdegree);
        // for (int i = in_start; i < in_end; i++) {
        //    vertex[tid]->setInEdge(i - in_start, incol[i], 0.0f);
        //}

        for (int i = out_start; i < out_end; i++) {
            vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i],
                                    INT_MAX);
        }
    }
}

__managed__ __align__(16) char buf2[128];
template <class myType>
__global__ void vptrPatch(myType *array, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("-----\n");
    myType *obj;
    obj = new (buf2) myType();
    // void *p;
    // p=(void *)0x111111111;
    // memcpy(p, obj, sizeof(void *));
    // printf("---%p--\n", p);
    if (tid < n) {
        memcpy(&array[tid], obj, sizeof(void *));
        // printf("---%p--\n",p);
    }
}

__global__ void vptrPatch_Edge(ChiVertex<int, int> *vertex, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Edge<int> *obj;
    obj = new (buf2) Edge<int>();

    if (tid < n)
        if (tid == 0)
            vertex[tid].vptrPatch(obj, 1);
        else
            vertex[tid].vptrPatch(obj, 1);
}

__managed__ obj_info_tuble *vfun_table;
__managed__ unsigned tree_size_g;
__managed__ void *temp_copyBack;
__managed__ void *temp_TP;

__global__ void BFS(ChiVertex<int, int> **vertex, GraphChiContext *context,
                    int iteration) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    void **vtable;
    if (tid < context->getNumVertices()) {
        if (iteration == 0) {
            if (tid == 0) {
                vertex[tid]->setValue(0);
                int numOutEdge;
                numOutEdge = vertex[tid]->numOutEdges();
                for (int i = 0; i < numOutEdge; i++) {
                    ChiEdge<int> *outEdge;
                    outEdge = vertex[tid]->getOutEdge(i);
                    vtable = get_vfunc_type(outEdge,vfun_table);
                    temp_TP = vtable[2];
                    CLEANPTR(outEdge, ChiEdge<int> *)->setValue(1);
                    //outEdge->setValue(1);
                }
            }
        } else {
            int curmin;
            curmin = vertex[tid]->getValue();
            int numInEdge;
            numInEdge = vertex[tid]->numInEdges();
            for (int i = 0; i < numInEdge; i++) {
                ChiEdge<int> *inEdge;
                inEdge = vertex[tid]->getInEdge(i);
                vtable = get_vfunc_type(inEdge,vfun_table);
                temp_TP = vtable[1];
                 curmin = min(curmin,  CLEANPTR(inEdge,ChiEdge<int> *)->getValue());
                //curmin = min(curmin,  inEdge->getValue());
            }
            int vertValue;
            vertValue = vertex[tid]->getValue();
            if (curmin < vertValue) {
                vertex[tid]->setValue(curmin);
                int numOutEdge;
                numOutEdge = vertex[tid]->numOutEdges();
                for (int i = 0; i < numOutEdge; i++) {
                    ChiEdge<int> *outEdge;
                    outEdge = vertex[tid]->getOutEdge(i);
                    int edgeValue;
                    vtable = get_vfunc_type(outEdge,vfun_table);
                    temp_TP = vtable[1];
                     edgeValue =  CLEANPTR(outEdge, ChiEdge<int> *)->getValue();
                    //edgeValue =  outEdge->getValue();
                    if (edgeValue > curmin + 1) {
                        vtable = get_vfunc_type(outEdge,vfun_table);
                        temp_TP = vtable[2];
                        CLEANPTR(outEdge, ChiEdge<int> *)->setValue(curmin + 1);
                    }
                }
            }
        }
    }
}

__global__ void copyBack(ChiVertex<int, int> **vertex, GraphChiContext *context,
                         int *index) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        index[tid] = vertex[tid]->getValue();
    }
}
