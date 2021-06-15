/*
__global__ void initContext(GraphChiContext* context, int vertices, int edges) {

        context->setNumIterations(0);
        context->setNumVertices(vertices);
        context->setNumEdges(edges);

}

__global__ void initObject(VirtVertex<int, int> **vertex, GraphChiContext*
context,
        int* row, int* col, int* inrow, int* incol) {
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
        vertex[tid] = new ChiVertex<int, int>(tid, indegree, outdegree);
        for (int i = in_start; i < in_end; i++) {
            CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setInEdge(i - in_start, incol[i], 0);
        }
        //for (int i = out_start; i < out_end; i++) {
        //    CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        //}
    }
}

__global__ void initOutEdge(VirtVertex<int, int> **vertex, GraphChiContext*
context,
        int* row, int* col) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        int out_start = row[tid];
        int out_end;
        if (tid + 1 < context->getNumVertices()) {
            out_end = row[tid + 1];
        } else {
            out_end = context->getNumEdges();
        }
        for (int i = out_start; i < out_end; i++) {
            CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setOutEdge(vertex, tid, i - out_start, col[i], 0);
        }
    }
}
*/
#include "parse_oo.h"

void initContext(GraphChiContext *context, int vertices, int edges) {
    // int tid = blockDim.x * blockIdx.x + threadIdx.x;

    context->setNumIterations(0);
    context->setNumVertices(vertices);
    context->setNumEdges(edges);
}

void part0_initObject(VirtVertex<int, int> **vertex, GraphChiContext *context,
                      int *row, int *col, int *inrow, int *incol,
                      obj_alloc *alloc) {
    int tid = 0;

    for (tid = 0; tid < context->getNumVertices(); tid++) {
        vertex[tid] =
            (VirtVertex<int, int> *)alloc->my_new<ChiVertex<int, int>>();
    }
}
void part1_initObject(VirtVertex<int, int> **vertex, GraphChiContext *context,
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
        CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->set_in_out(alloc);
        // vertex[tid].setValue(INT_MAX);
        // for (int i = in_start; i < in_end; i++) {
        //   vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
        // }
        // for (int i = out_start; i < out_end; i++) {
        //    CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        //}
    }
}
__global__ void part_kern0_initObject(VirtVertex<int, int> **vertex,
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
        new (CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *))
            ChiVertex<int, int>(tid, indegree, outdegree);

        // for (int i = out_start; i < out_end; i++) {
        //    CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        //}
    }

    // vertex[tid].setValue(INT_MAX);
    // for (int i = in_start; i < in_end; i++) {
    //   vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
    // }
}
__global__ void part_kern1_initObject(VirtVertex<int, int> **vertex,
                                      GraphChiContext *context, int *row,
                                      int *col, int *inrow, int *incol) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < context->getNumVertices()) {
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

        for (int i = in_start; i < in_end; i++) {
            CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)
                ->setInEdge(i - in_start, incol[i], 0);
        }
    }
    // for (int i = out_start; i < out_end; i++) {
    //    CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
    //}
}

__global__ void kern_initOutEdge(VirtVertex<int, int> **vertex,
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
        for (int i = out_start; i < out_end; i++) {
            CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)
                ->setOutEdge(vertex, tid, i - out_start, col[i], 0);
        }
    }
}

__managed__ obj_info_tuble *vfun_table;
__managed__ unsigned tree_size_g;
__managed__ void *temp_copyBack;
__managed__ void *temp_TP;

__global__ void ConnectedComponent(VirtVertex<int, int> **vertex,
                                   GraphChiContext *context, int iteration) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    void **vtable;
    if (tid < context->getNumVertices()) {
        int numEdges;
        vtable = get_vfunc_type(vertex[tid], vfun_table);;
        temp_TP = vtable[4];
        numEdges = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->numEdges();
        if (iteration == 0) {
            vtable = get_vfunc_type(vertex[tid], vfun_table);;
            temp_TP = vtable[0];
            int vid = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->getId();
            vtable = get_vfunc_type(vertex[tid], vfun_table);;
            temp_TP = vtable[3];
            CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setValue(vid);
        }
        int curMin;
        vtable = get_vfunc_type(vertex[tid], vfun_table);;
        temp_TP = vtable[2];
        curMin = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->getValue();
        for (int i = 0; i < numEdges; i++) {
            ChiEdge<int> *edge;
            vtable = get_vfunc_type(vertex[tid], vfun_table);;
            temp_TP = vtable[11];
            edge = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->edge(i);
            int nbLabel;
            vtable = get_vfunc_type(edge, vfun_table);
            temp_TP = vtable[1];
            nbLabel = CLEANPTR(edge, ChiEdge<int> *)->getValue();
            if (iteration == 0) {
                vtable = get_vfunc_type(edge, vfun_table);
                temp_TP = vtable[0];
                nbLabel = CLEANPTR(edge, ChiEdge<int> *)->getVertexId();  // Note!
            }
            if (nbLabel < curMin) {
                curMin = nbLabel;
            }
        }

        /**
         * Set my new label
         */
        vtable = get_vfunc_type(vertex[tid], vfun_table);;
        temp_TP = vtable[3];
        CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->setValue(curMin);
        int label = curMin;

        /**
         * Broadcast my value to neighbors by writing the value to my edges.
         */
        if (iteration > 0) {
            for (int i = 0; i < numEdges; i++) {
                ChiEdge<int> *edge;
                vtable = get_vfunc_type(vertex[tid], vfun_table);;
                temp_TP = vtable[11];
                edge = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->edge(i);
                int edgeValue;
                vtable = get_vfunc_type(edge, vfun_table);
                temp_TP = vtable[1];
                edgeValue = CLEANPTR(edge, ChiEdge<int> *)->getValue();
                if (edgeValue > label) {
                    vtable = get_vfunc_type(edge, vfun_table);
                    temp_TP = vtable[2];
                    CLEANPTR(edge, ChiEdge<int> *)->setValue(label);
                }
            }
        } else {
            // Special case for first iteration to avoid overwriting
            int numOutEdge;
            vtable = get_vfunc_type(vertex[tid], vfun_table);;
            temp_TP = vtable[6];
            numOutEdge = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->numOutEdges();
            for (int i = 0; i < numOutEdge; i++) {
                ChiEdge<int> *outEdge;
                vtable = get_vfunc_type(vertex[tid], vfun_table);;
                temp_TP = vtable[8];
                outEdge = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->getOutEdge(i);
                vtable = get_vfunc_type(outEdge, vfun_table);
                temp_TP = vtable[2];
                CLEANPTR(outEdge, ChiEdge<int> *)->setValue(label);
            }
        }
    }
}

__global__ void copyBack(VirtVertex<int, int> **vertex,
                         GraphChiContext *context, int *cc) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    void **vtable;

    if (tid < context->getNumVertices()) {
        vtable = get_vfunc_type(vertex[tid], vfun_table);;
        temp_TP = vtable[1];
        cc[tid] = CLEANPTR(vertex[tid], VirtVertex<int COMMA int> *)->getValue();
    }
}
