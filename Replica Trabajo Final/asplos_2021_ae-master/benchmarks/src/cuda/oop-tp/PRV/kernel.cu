#include "parse_oo.h"

void initContext(GraphChiContext *context, int vertices, int edges) {
    context->setNumIterations(0);
    context->setNumVertices(vertices);
    context->setNumEdges(edges);
}
void part0_initObject(VirtVertex<float, float> **vertex,
                      GraphChiContext *context, int *row, int *col, int *inrow,
                      int *incol, obj_alloc *alloc) {
    int tid = 0;

    for (tid = 0; tid < context->getNumVertices(); tid++) {
        vertex[tid] = (VirtVertex<float, float> *)
                          alloc->my_new<ChiVertex<float, float>>();
    }
}
void part1_initObject(VirtVertex<float, float> **vertex,
                      GraphChiContext *context, int *row, int *col, int *inrow,
                      int *incol, obj_alloc *alloc) {
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
        CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)
            ->set_in_out(alloc);
        // vertex[tid].setValue(INT_MAX);
        // for (int i = in_start; i < in_end; i++) {
        //   vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
        // }
        // for (int i = out_start; i < out_end; i++) {
        //    CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        //}
    }
}

__global__ void part_kern0_initObject(VirtVertex<float, float> **vertex,
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
        new (CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *))
            ChiVertex<float, float>(tid, indegree, outdegree);
    }
}
__global__ void part_kern1_initObject(VirtVertex<float, float> **vertex,
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

        for (int i = in_start; i < in_end; i++) {
            CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)
                ->setInEdge(i - in_start, incol[i], 0.0f);
        }
    }
}
__global__ void kern_initOutEdge(VirtVertex<float, float> **vertex,
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
        //    CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->setInEdge(i - in_start, incol[i], 0.0f);
        //}

        for (int i = out_start; i < out_end; i++) {
            CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)
                ->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        }
    }
}

__managed__ obj_info_tuble *vfun_table;
__managed__ unsigned tree_size_g;
__managed__ void *temp_copyBack;
__managed__ void *temp_TP;

__global__ void PageRank(VirtVertex<float, float> **vertex,
                         GraphChiContext *context, int iteration) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;


    void **vtable;
    if (tid < context->getNumVertices()) {
        if (iteration == 0) {
            vtable = get_vfunc_type(vertex[tid], vfun_table);
            temp_TP = vtable[3];
            CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->setValue(1.0f);
        } else {
            float sum = 0.0f;
            int numInEdge;
            vtable = get_vfunc_type(vertex[tid], vfun_table);
            temp_TP = vtable[4];
            numInEdge = CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->numInEdges();
            for (int i = 0; i < numInEdge; i++) {
                ChiEdge<float> *inEdge;
                vtable = get_vfunc_type(vertex[tid], vfun_table);
                temp_TP = vtable[6];
                inEdge = CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->getInEdge(i);
                vtable = get_vfunc_type(inEdge, vfun_table);
                temp_TP = vtable[1];
                sum += CLEANPTR(inEdge, ChiEdge<float> *)->getValue();
            }
            vtable = get_vfunc_type(vertex[tid], vfun_table);
            temp_TP = vtable[3];
            CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->setValue(0.15f + 0.85f * sum);

            /* Write my value (divided by my out-degree) to my out-edges so
             * neighbors can read it. */
            int numOutEdge;
            vtable = get_vfunc_type(vertex[tid], vfun_table);
            temp_TP = vtable[5];
            numOutEdge = CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->numOutEdges();
            vtable = get_vfunc_type(vertex[tid], vfun_table);
            temp_TP = vtable[2];
            float outValue = CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->getValue() / numOutEdge;
            for (int i = 0; i < numOutEdge; i++) {
                ChiEdge<float> *outEdge;
                vtable = get_vfunc_type(vertex[tid], vfun_table);
                temp_TP = vtable[7];
                outEdge = CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->getOutEdge(i);
                vtable = get_vfunc_type(outEdge, vfun_table);
                temp_TP = vtable[2];
                CLEANPTR(outEdge, ChiEdge<float> *)->setValue(outValue);
            }
        }
    }
}

__global__ void copyBack(VirtVertex<float, float> **vertex,
                         GraphChiContext *context, float *pagerank) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        void **vtable = get_vfunc_type(vertex[tid], vfun_table);
        temp_TP = vtable[2];
        pagerank[tid] =
            CLEANPTR(vertex[tid], VirtVertex<float COMMA float> *)->getValue();
    }
}
