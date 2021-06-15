#include "parse_oo.h"

__global__ void initContext(GraphChiContext* context, int vertices, int edges) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
        context->setNumIterations(0);
        context->setNumVertices(vertices);
        context->setNumEdges(edges);
    }
}

__global__ void initObject(ChiVertex<float, float>** vertex,
                           GraphChiContext* context, int* row, int* col,
                           int* inrow, int* incol) {
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
        vertex[tid] = new ChiVertex<float, float>(tid, indegree, outdegree);
        for (int i = in_start; i < in_end; i++) {
            vertex[tid]->setInEdge(i - in_start, incol[i], 0.0f);
        }
        // for (int i = out_start; i < out_end; i++) {
        //    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        //}
    }
}

__global__ void initOutEdge(ChiVertex<float, float>** vertex,
                            GraphChiContext* context, int* row, int* col) {
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
            vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
        }
    }
}

__global__ void PageRank(ChiVertex<float, float>** vertex,
                         GraphChiContext* context, int iteration) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        if (iteration == 0) {
            vertex[tid]->setValue(1.0f);
        } else {
            float sum = 0.0f;
            int numInEdge;
            numInEdge = vertex[tid]->numInEdges();
            for (int i = 0; i < numInEdge; i++) {
                ChiEdge<float>* inEdge;
                inEdge = vertex[tid]->getInEdge(i);
                sum += inEdge->getValue();
            }
            vertex[tid]->setValue(0.15f + 0.85f * sum);

            /* Write my value (divided by my out-degree) to my out-edges so
             * neighbors can read it. */
            int numOutEdge;
            numOutEdge = vertex[tid]->numOutEdges();
            float outValue = vertex[tid]->getValue() / numOutEdge;
            for (int i = 0; i < numOutEdge; i++) {
                ChiEdge<float>* outEdge;
                outEdge = vertex[tid]->getOutEdge(i);
                outEdge->setValue(outValue);
            }
        }
    }
}

__global__ void copyBack(ChiVertex<float, float>** vertex,
                         GraphChiContext* context, float* pagerank) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < context->getNumVertices()) {
        pagerank[tid] = vertex[tid]->getValue();
    }
}
