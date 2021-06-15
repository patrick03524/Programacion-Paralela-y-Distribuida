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
    for (int i = in_start; i < in_end; i++) {
      vertex[tid]->setInEdgeV(i - in_start, incol[i], 0);
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
    for (int i = out_start; i < out_end; i++) {
      vertex[tid]->setOutEdgeV(vertex, tid, i - out_start, col[i], 0);
    }
  }
}

__global__ void ConnectedComponent(VirtVertex<int, int> **vertex,
                                   GraphChiContext *context, int iteration) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < context->getNumVertices()) {
    int numEdges;

    switch (vertex[tid]->type) {
    case 0:
      numEdges = vertex[tid]->numEdgesC();
      break;
    case 1:
      numEdges = vertex[tid]->numEdgesV();
      break;
    }
    if (iteration == 0) {
      int vid;

      switch (vertex[tid]->type) {
      case 0:
        vid = vertex[tid]->getIdC();
        break;
      case 1:
        vid = vertex[tid]->getIdV();
        break;
      }

      switch (vertex[tid]->type) {
      case 0:
        vertex[tid]->setValueC(vid);
        break;
      case 1:
        vertex[tid]->setValueV(vid);
        break;
      }
    }
    int curMin;

    switch (vertex[tid]->type) {
    case 0:
      curMin = vertex[tid]->getValueC();
      break;
    case 1:
      curMin = vertex[tid]->getValueV();
      break;
    }
    for (int i = 0; i < numEdges; i++) {
      ChiEdge<int> *edge;

      switch (vertex[tid]->type) {
      case 0:
        edge = vertex[tid]->edgeC(i);
        break;
      case 1:
        edge = vertex[tid]->edgeV(i);
        break;
      }
      int nbLabel;

      switch (edge->type) {
      case 0:
        nbLabel = edge->getValueC();
        break;
      case 1:
        nbLabel = edge->getValueV();
        break;
      }
      if (iteration == 0) {
        switch (edge->type) {
        case 0:
          nbLabel = edge->getVertexIdC(); // Note!
          break;
        case 1:
          nbLabel = edge->getVertexIdV(); // Note!
          break;
        }
      }
      if (nbLabel < curMin) {
        curMin = nbLabel;
      }
    }

    /**
     * Set my new label
     */

    switch (vertex[tid]->type) {
    case 0:
      vertex[tid]->setValueC(curMin);
      break;
    case 1:
      vertex[tid]->setValueV(curMin);
      break;
    }
    int label = curMin;

    /**
     * Broadcast my value to neighbors by writing the value to my edges.
     */
    if (iteration > 0) {
      for (int i = 0; i < numEdges; i++) {
        ChiEdge<int> *edge;

        switch (vertex[tid]->type) {
        case 0:
          edge = vertex[tid]->edgeC(i);
          break;
        case 1:
          edge = vertex[tid]->edgeV(i);
          break;
        }
        int edgeValue;

        switch (edge->type) {
        case 0:
          edgeValue = edge->getValueC();
          break;
        case 1:
          edgeValue = edge->getValueV();
          break;
        }
        if (edgeValue > label) {
          switch (edge->type) {
          case 0:
            edge->setValueC(label);
            break;
          case 1:
            edge->setValueV(label);
            break;
          }
        }
      }
    } else {
      // Special case for first iteration to avoid overwriting
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
          outEdge->setValueC(label);
          break;
        case 1:
          outEdge->setValueV(label);
          break;
        }
      }
    }
  }
}

__global__ void copyBack(VirtVertex<int, int> **vertex,
                         GraphChiContext *context, int *cc) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < context->getNumVertices()) {
      
    cc[tid] = vertex[tid]->getValueV();
  }
}
