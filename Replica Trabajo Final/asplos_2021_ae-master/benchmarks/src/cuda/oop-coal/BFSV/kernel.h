void initContext(GraphChiContext *context, int vertices, int edges) {
  // int tid = blockDim.x * blockIdx.x + threadIdx.x;

  context->setNumIterations(0);
  context->setNumVertices(vertices);
  context->setNumEdges(edges);
}

void initObject(VirtVertex<int, int> *vertex, GraphChiContext *context,
                int *row, int *col, int *inrow, int *incol, obj_alloc *alloc) {
  int tid = 0;

  for (tid = 0; tid < context->getNumVertices(); tid++) {
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
    // vertex[tid].inEdgeDataArray =
    //     (ChiEdge<int> *)alloc->my_new<Edge<int>>(indegree);
    // vertex[tid].outEdgeDataArray =
    //     (ChiEdge<int> **)alloc->my_new<Edge<int> *>(outdegree);
    new (&vertex[tid]) ChiVertex<int, int>(tid, indegree, outdegree, alloc);

    vertex[tid].setValue(INT_MAX);
    for (int i = in_start; i < in_end; i++) {
      vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
    }
    // for (int i = out_start; i < out_end; i++) {
    //    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
    //}
  }
}

void part0_initObject(VirtVertex<int, int> **vertex, GraphChiContext *context,
                     int *row, int *col, int *inrow, int *incol,
                     obj_alloc *alloc) {
  int tid = 0;

  for (tid = 0; tid < context->getNumVertices(); tid++) {
    
    vertex[tid]=(VirtVertex<int, int> *)alloc->my_new<ChiVertex<int, int>>();

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
    // new (&vertex[tid]) ChiVertex<int, int>(tid, indegree, outdegree,alloc);
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

    new (vertex[tid]) ChiVertex<int, int>(tid, indegree, outdegree);

    // vertex[tid].setValue(INT_MAX);
    // for (int i = in_start; i < in_end; i++) {
    //   vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
    // }
  }
}
__global__ void part_kern1_initObject(VirtVertex<int, int> **vertex,
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
void initOutEdge(VirtVertex<int, int> **vertex, GraphChiContext *context,
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
      vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], INT_MAX);
    }
  }
}

__global__ void kern_initObject(VirtVertex<int, int> *vertex,
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
    new (&vertex[tid]) ChiVertex<int, int>(tid, indegree, outdegree);

    vertex[tid].setValue(INT_MAX);
    for (int i = in_start; i < in_end; i++) {
      vertex[tid].setInEdge(i - in_start, incol[i], INT_MAX);
    }

    // for (int i = out_start; i < out_end; i++) {
    //    vertex[tid]->setOutEdge(vertex, tid, i - out_start, col[i], 0.0f);
  }
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

__managed__ __align__(16) char buf2[128];
template <class myType> __global__ void vptrPatch(myType *array, int n) {

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
__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size_g;
__managed__ void *temp_coal;



__global__ void BFS(VirtVertex<int, int> **vertex, GraphChiContext *context,int iteration) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned tree_size = tree_size_g;
  range_tree_node *table=range_tree;
  //  extern __shared__ range_tree_node table[];
  // if (threadIdx.x < tree_size) {

  //   //for (int i = 0; i < tree_size; i++) {
  //     //printf("%d\n",threadIdx.x);
  //     memcpy(&table[threadIdx.x], &range_tree[threadIdx.x], sizeof(range_tree_node));
  //     // if(tid==0)
  //     // printf("%p %p \n",table[i].range_start,table[i].range_end);
  // //  }
  // }
  // __syncthreads();
  void **vtable;
  void **vtable2;
  if (tid < context->getNumVertices()) {
    if (iteration == 0) {
      if (tid == 0) {
        // vtable = get_vfunc(vertex[tid], table, tree_size);
        // temp_coal = vtable[3];
        vertex[tid]->setValue(0);
        int numOutEdge;
        vtable = get_vfunc(vertex[tid], table, tree_size);
        temp_coal = vtable[5];

        numOutEdge = vertex[tid]->numOutEdges();
        for (int i = 0; i < numOutEdge; i++) {
          ChiEdge<int> *outEdge;
          vtable = get_vfunc(vertex[tid], table, tree_size);
          temp_coal = vtable[7];
          outEdge = vertex[tid]->getOutEdge(i);
          vtable2 = get_vfunc(outEdge, table, tree_size);
          temp_coal = vtable2[2];

          outEdge->setValue(1);
        }
      }
    } else {
      int curmin;
      vtable = get_vfunc(vertex[tid], table, tree_size);
      temp_coal = vtable[2];
      curmin = vertex[tid]->getValue();
      int numInEdge;
      vtable = get_vfunc(vertex[tid], table, tree_size);
      temp_coal = vtable[4];
      numInEdge = vertex[tid]->numInEdges();
      for (int i = 0; i < numInEdge; i++) {
        ChiEdge<int> *inEdge;
        vtable = get_vfunc(vertex[tid], table, tree_size);
        temp_coal = vtable[6];
        inEdge = vertex[tid]->getInEdge(i);
        vtable2 = get_vfunc(inEdge, table, tree_size);
        temp_coal = vtable2[1];
        curmin = min(curmin, inEdge->getValue());
      }
      int vertValue;
      vtable = get_vfunc(vertex[tid], table, tree_size);
      temp_coal = vtable[2];
      vertValue = vertex[tid]->getValue();
      if (curmin < vertValue) {
        vtable = get_vfunc(vertex[tid], table, tree_size);
        temp_coal = vtable[3];
        vertex[tid]->setValue(curmin);
        int numOutEdge;
        vtable = get_vfunc(vertex[tid], table, tree_size);
        temp_coal = vtable[5];
        numOutEdge = vertex[tid]->numOutEdges();
        for (int i = 0; i < numOutEdge; i++) {
          ChiEdge<int> *outEdge;
          vtable = get_vfunc(vertex[tid], table, tree_size);
          temp_coal = vtable[7];
          outEdge = vertex[tid]->getOutEdge(i);
          int edgeValue;
          vtable2 = get_vfunc(outEdge, table, tree_size);
          temp_coal = vtable2[1];
          edgeValue = outEdge->getValue();
          if (edgeValue > curmin + 1) {
            temp_coal = vtable2[2];
            outEdge->setValue(curmin + 1);
          }
        }
      }
    }
   
  }
}
__managed__ void *temp_vfun;
__global__ void vfunCheck(VirtVertex<int, int> *vertex) {
  void **vtable;
  unsigned tree_size = tree_size_g;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  vtable = get_vfunc(&vertex[tid], range_tree, tree_size);
  temp_vfun = vtable[1];
  vertex[tid].setId(155);
  temp_vfun = vtable[0];
  printf("%d\n", vertex[tid].getId());

  temp_vfun = vtable[3];
  vertex[tid].setValue(999);
  temp_vfun = vtable[2];
  printf("%d\n", vertex[tid].getValue());
  temp_vfun = vtable[4];
  printf("%d\n", vertex[tid].numInEdges());
  temp_vfun = vtable[5];
  printf("%d\n", vertex[tid].numOutEdges());
  temp_vfun = vtable[6];
  printf("%p\n", vertex[tid].getInEdge(0));
  temp_vfun = vtable[7];
  printf("%p\n", vertex[tid].getOutEdge(0));
}

void BFS_cpu(VirtVertex<int, int> *vertex, GraphChiContext *context) {
  int tid = 0;
  // printf("ffff\n");
  for (tid = 0; tid < context->getNumVertices(); tid++) {
    if (context->getNumIterations() == 0) {
      if (tid == 0) {

        vertex[tid].setValue(0);
        int numOutEdge;
        numOutEdge = vertex[tid].numOutEdges();
        for (int i = 0; i < numOutEdge; i++) {
          ChiEdge<int> *outEdge;
          outEdge = vertex[tid].getOutEdge(i);
          outEdge->setValue(1);
        }
      }
    } else {
      int curmin;
      curmin = vertex[tid].getValue();
      int numInEdge;
      numInEdge = vertex[tid].numInEdges();
      for (int i = 0; i < numInEdge; i++) {
        ChiEdge<int> *inEdge;
        inEdge = vertex[tid].getInEdge(i);
        curmin = min(curmin, inEdge->getValue());
      }
      int vertValue;
      vertValue = vertex[tid].getValue();
      if (curmin < vertValue) {
        vertex[tid].setValue(curmin);
        int numOutEdge;
        numOutEdge = vertex[tid].numOutEdges();
        for (int i = 0; i < numOutEdge; i++) {
          ChiEdge<int> *outEdge;
          outEdge = vertex[tid].getOutEdge(i);
          int edgeValue;
          edgeValue = outEdge->getValue();
          if (edgeValue > curmin + 1) {
            outEdge->setValue(curmin + 1);
          }
        }
      }
    }
    //  context->setNumIterations(context->getNumIterations() + 1);
  }
}

__global__ void copyBack(VirtVertex<int, int> **vertex, GraphChiContext *context,
                         int *index) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned tree_size = tree_size_g;
  // ChiVertex<int, int> *obj;
  // obj = new (buf2) ChiVertex<int, int>();
  // long ***mVtable = (long ***)&vertex[tid];
  // long ***mVtable2 = (long ***)obj;
  // //memcpy(&vertex[tid],obj,sizeof(void*));
  // printf("[%d]-obj %p vert %p\n",tid,*mVtable2,*mVtable);
  // *mVtable=*mVtable2;
  // printf("[%d]after obj %p vert %p\n",tid,*mVtable2,*mVtable);
  if (tid < context->getNumVertices()) {
    void **vtable = get_vfunc(vertex[tid], range_tree, tree_size);
    temp_coal = vtable[2];
    // printf("%d\n",index[tid]);
    index[tid] = vertex[tid]->getValue();
    //  if(mVtable[0][0]!=mVtable2[0][0])
    //  printf("[%d]why !! obj %p vert %p\n",tid,mVtable[0][0],mVtable2[0][0]);
    // printf("%d\n",index[tid]);
  }
}