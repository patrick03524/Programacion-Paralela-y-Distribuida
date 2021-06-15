
#define ALL __noinline__ __host__ __device__
template <typename EdgeData>
class ChiEdge {
  public:
    ALL virtual int getVertexId() = 0;
    ALL virtual EdgeData getValue() = 0;
    ALL virtual void setValue(EdgeData x) = 0;
    ALL ChiEdge() {}
    ALL ChiEdge(int id, int value) {}
    EdgeData edgeValue;
    int vertexId;
};

template <typename EdgeValue>
class Edge : public ChiEdge<EdgeValue> {
  public:
    ALL Edge() {}
    ALL Edge(int id, int value) : ChiEdge<EdgeValue>(id, value) {
        this->vertexId = id;
        this->edgeValue = value;
    }
    ALL int getVertexId() {
        // if (print)
        //   printf("getVertexId\n");
        return this->vertexId;
    }
    ALL EdgeValue getValue() {
        // if (print)
        //   printf("getValue\n");
        return this->edgeValue;
    }
    ALL void setValue(EdgeValue x) {
        // if (print)
        //   printf("setValue\n");
        this->edgeValue = x;
    }
};

template <typename VertexValue, typename EdgeValue>
class ChiVertex {
  public:
    // init functions
    ALL ChiVertex() {}
    ChiVertex(int id, int inDegree, int outDegree, obj_alloc *alloc) {
        this->id = id;
        this->nInedges = inDegree;
        this->nOutedges = outDegree;
        // this->inEdgeDataArray =
        //     (ChiEdge<EdgeValue> **)alloc->my_new<Edge<EdgeValue>>(inDegree);
        // this->outEdgeDataArray =
        //     (ChiEdge<EdgeValue> **)alloc->my_new<Edge<EdgeValue>>(outDegree);
    }
    void set_in_out(obj_alloc *alloc) {
        this->inEdgeDataArray =
            (ChiEdge<EdgeValue> **)alloc->calloc<ChiEdge<EdgeValue> *>(
                this->nInedges);
        this->outEdgeDataArray =
            (ChiEdge<EdgeValue> **)alloc->calloc<ChiEdge<EdgeValue> *>(
                this->nOutedges);
        for (int i = 0; i < this->nInedges; i++) {
            this->inEdgeDataArray[i] =
                (Edge<EdgeValue> *)alloc->my_new<Edge<EdgeValue>>();
        }
    }
    ALL ChiVertex(int id, int inDegree, int outDegree) {
        this->id = id;
        this->nInedges = inDegree;
        this->nOutedges = outDegree;
        // this->inEdgeDataArray =
        //     (ChiEdge<EdgeValue> *)malloc(sizeof(ChiEdge<EdgeValue>
        //     )*inDegree);
        // this->outEdgeDataArray =
        //     (ChiEdge<EdgeValue> **)malloc(sizeof(ChiEdge<EdgeValue>
        //     *)*outDegree);
    }
    // operation functions
    ALL int getId() {
        // if (print)
        //   printf("getID\n");
        return this->id;
    }
    ALL void setId(int x) { this->id = x; }
    ALL VertexValue getValue() {
        // if (print)
        //   printf("getValue\n");
        return this->value;
    }
    ALL void setValue(VertexValue x) {
        // if (print)
        // printf("setValue %d \n",x);
        this->value = x;
    }
    ALL int numInEdges() {
        // if (print)
        //   printf("numInEdges\n");
        return this->nInedges;
    }
    ALL int numOutEdges() {
        // if (print)
        //   printf("numOutEdges\n");
        return this->nOutedges;
    }
    ALL ChiEdge<EdgeValue> *getInEdge(int i) {
        // if (print)
        //   printf("getInEdge\n");
        return this->inEdgeDataArray[i];
    }
    ALL ChiEdge<EdgeValue> *getOutEdge(int i) {
        // if (print)
        //   printf("getOutEdge\n");
        return this->outEdgeDataArray[i];
    }
    ALL void setInEdge(int idx, int vertexId, EdgeValue value) {
        // if (print)
        //   printf("setInEdge\n");
        new (this->inEdgeDataArray[idx]) Edge<EdgeValue>(vertexId, value);
    }
    ALL void vptrPatch(Edge<EdgeValue> *gpu_obj, int test) {
        int i;
        // printf("ggg\n");
        for (i = 0; i < this->nInedges; i++) {
            if (test == 0) {
                long ***mVtable = (long ***)&this->inEdgeDataArray[i];
                printf("Derived VTABLE before: %p %p\n",
                       &this->inEdgeDataArray[i], *mVtable);
                memcpy(&this->inEdgeDataArray[i], gpu_obj, sizeof(void *));
                printf("Derived VTABLE after: %p %p\n",
                       &this->inEdgeDataArray[i], *mVtable);
            }
            memcpy(&this->inEdgeDataArray[i], gpu_obj, sizeof(void *));
        }
    }
    ALL void setOutEdge(ChiVertex<int, int> **vertex, int src, int idx,
                        int vertexId, EdgeValue value) {
        // if (print)
        //   printf("setOutEdge\n");
        // outEdgeDataArray[idx] = new Edge<EdgeValue>(vertexId, value);
        for (int i = 0; i < vertex[vertexId]->numInEdges(); i++) {
            if (vertex[vertexId]->getInEdge(i)->getVertexId() == src) {
                this->outEdgeDataArray[idx] = vertex[vertexId]->getInEdge(i);
                break;
            }
        }
    }

  private:
    int id;
    int nInedges;
    int nOutedges;
    VertexValue value;
    ChiEdge<EdgeValue> **inEdgeDataArray;
    ChiEdge<EdgeValue> **outEdgeDataArray;
};

class GraphChiContext {
  public:
    ALL GraphChiContext(int vert) {}
    ALL int getNumIterations() { return numIterations; }

    ALL void setNumIterations(int iter) { numIterations = iter; }
    ALL int getNumVertices() { return numVertices; }

    ALL void setNumVertices(int vertices) { numVertices = vertices; }
    ALL int getNumEdges() { return numEdges; }

    ALL void setNumEdges(int edges) { numEdges = edges; }

  private:
    int numIterations;
    int numVertices;
    int numEdges;
};