__global__ void initContext(GraphChiContext *context, int vertices, int edges) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    context->setNumIterations(0);
    context->setNumVertices(vertices);
    context->setNumEdges(edges);
  }
}

__global__ void initObject(VirtVertex<int, int> **vertex,
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
    vertex[tid]->setValueV(INT_MAX);
    for (int i = in_start; i < in_end; i++) {
      vertex[tid]->setInEdgeV(i - in_start, incol[i], INT_MAX);
    }
    // for (int i = out_start; i < out_end; i++) {
    //    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
    //}
  }
}

__global__ void initOutEdge(VirtVertex<int, int> **vertex,
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
      vertex[tid]->setOutEdgeV(vertex, tid, i - out_start, col[i], INT_MAX);
    }
  }
}

__global__ void BFS(VirtVertex<int, int> **vertex, GraphChiContext *context,
                    int iteration) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < context->getNumVertices()) {
    if (iteration == 0) {
      if (tid == 0) {

        switch (vertex[tid]->type) {
        case 0:
          vertex[tid]->setValueC(0);
          break;
        case 1:
          vertex[tid]->setValueV(0);
          break;
        }
        int numOutEdge;

        switch (vertex[tid]->type) {
        case 0:
          numOutEdge = vertex[tid]->numOutEdgesC();
          break;
        case 1:
          numOutEdge = vertex[tid]->numOutEdgesV();
          break;
        }
        for (int i = 0; i < numOutEdge; i++) {
          ChiEdge<int> *outEdge;

          switch (vertex[tid]->type) {
          case 0:
            outEdge = vertex[tid]->getOutEdgeC(i);
            break;
          case 1:
            outEdge = vertex[tid]->getOutEdgeV(i);
            break;
          }
          switch (outEdge->type) {
          case 0:
            outEdge->setValueC(1);
            break;
          case 1:
            outEdge->setValueV(1);
            break;
          }
        }
      }
    } else {
      int curmin;

      switch (vertex[tid]->type) {
      case 0:
        curmin = vertex[tid]->getValueC();
        break;
      case 1:
        curmin = vertex[tid]->getValueV();
        break;
      }
      int numInEdge;

      switch (vertex[tid]->type) {
      case 0:
        numInEdge = vertex[tid]->numInEdgesC();
        break;
      case 1:
        numInEdge = vertex[tid]->numInEdgesV();
        break;
      }
      for (int i = 0; i < numInEdge; i++) {
        ChiEdge<int> *inEdge;

        switch (vertex[tid]->type) {
        case 0:
          inEdge = vertex[tid]->getInEdgeC(i);
          break;
        case 1:
          inEdge = vertex[tid]->getInEdgeV(i);
          break;
        }
        switch (inEdge->type) {
        case 0:
          curmin = min(curmin, inEdge->getValueC());
          break;
        case 1:
          curmin = min(curmin, inEdge->getValueV());
          break;
        }
      }
      int vertValue;

      switch (vertex[tid]->type) {
      case 0:
        vertValue = vertex[tid]->getValueC();
        break;
      case 1:
        vertValue = vertex[tid]->getValueV();
        break;
      }
      if (curmin < vertValue) {

        switch (vertex[tid]->type) {
        case 0:
          vertex[tid]->setValueC(curmin);
          break;
        case 1:
          vertex[tid]->setValueV(curmin);
          break;
        }
        int numOutEdge;
        switch (vertex[tid]->type) {
        case 0:
          numOutEdge = vertex[tid]->numOutEdgesC();
          break;
        case 1:
          numOutEdge = vertex[tid]->numOutEdgesV();
          break;
        }

        for (int i = 0; i < numOutEdge; i++) {
          ChiEdge<int> *outEdge;

          switch (vertex[tid]->type) {
          case 0:
            outEdge = vertex[tid]->getOutEdgeC(i);
            break;
          case 1:
            outEdge = vertex[tid]->getOutEdgeV(i);
            break;
          }

          int edgeValue;

          switch (outEdge->type) {
          case 0:
            edgeValue = outEdge->getValueC();
            break;
          case 1:
            edgeValue = outEdge->getValueV();
            break;
          }
          if (edgeValue > curmin + 1) {
            switch (outEdge->type) {
            case 0:
              outEdge->setValueC(curmin + 1);
              break;
            case 1:
              outEdge->setValueV(curmin + 1);
              break;
            }
          }
        }
      }
    }
  }
}

__global__ void copyBack(VirtVertex<int, int> **vertex,
                         GraphChiContext *context, int *index) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < context->getNumVertices()) {

    index[tid] = vertex[tid]->getValueV();
  }
}
