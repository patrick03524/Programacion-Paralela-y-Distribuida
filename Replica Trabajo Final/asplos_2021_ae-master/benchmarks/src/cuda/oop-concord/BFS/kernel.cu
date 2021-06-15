__global__ void initContext(GraphChiContext *context, int vertices, int edges) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    context->setNumIterations(0);
    context->setNumVertices(vertices);
    context->setNumEdges(edges);
  }
}

__global__ void initObject(ChiVertex<int, int> **vertex,
                           GraphChiContext *context, int *row, int *col,
                           int *inrow, int *incol) {
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
    vertex[tid]->setValue(INT_MAX);
    for (int i = in_start; i < in_end; i++) {
      vertex[tid]->setInEdge(i - in_start, incol[i], INT_MAX);
    }
    // for (int i = out_start; i < out_end; i++) {
    //    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
    //}
  }
}

__global__ void initOutEdge(ChiVertex<int, int> **vertex,
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
      vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], INT_MAX);
    }
  }
}

__global__ void BFS(ChiVertex<int, int> **vertex, GraphChiContext *context,
                    int iteration) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < context->getNumVertices()) {
    if (iteration == 0) {
      if (tid == 0) {
        vertex[tid]->setValue(0);
        int numOutEdge;
        numOutEdge = vertex[tid]->numOutEdges();
        for (int i = 0; i < numOutEdge; i++) {
          ChiEdge<int> *outEdge;
          outEdge = vertex[tid]->getOutEdge(i);
          switch (outEdge->type) {
          case 0:
            outEdge->setValueChiEdge(1);
            break;
          case 1:
            outEdge->setValueEdge(1);
            break;
          }
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
        switch (inEdge->type) {
        case 0:
          curmin = min(curmin, inEdge->getValueChiEdge());
          break;
        case 1:
          curmin = min(curmin, inEdge->getValueEdge());
          break;
        }
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
          switch (outEdge->type) {
            case 0:
            edgeValue = outEdge->getValueChiEdge();
              break;
            case 1:
            edgeValue = outEdge->getValueEdge();
              break;
            }
         
          if (edgeValue > curmin + 1) {
            switch (outEdge->type) {
            case 0:
              outEdge->setValueChiEdge(curmin + 1);
              break;
            case 1:
              outEdge->setValueEdge(curmin + 1);
              break;
            }
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
