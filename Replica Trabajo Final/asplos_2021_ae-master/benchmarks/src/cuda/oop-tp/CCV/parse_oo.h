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
    ALL Edge(int id, int value) {
        this->vertexId = id;
        this->edgeValue = value;
    }
    ALL int getVertexId() { return this->vertexId; }
    ALL EdgeValue getValue() { return this->edgeValue; }
    ALL void setValue(EdgeValue x) { this->edgeValue = x; }
};

template <typename VertexValue, typename EdgeValue>
class ChiVertex;

template <typename VertexValue, typename EdgeValue>
class VirtVertex {
  public:
    // operation functions
    ALL virtual int getId() = 0;
    ALL virtual void setId(int x) = 0;
    ALL virtual VertexValue getValue() = 0;
    ALL virtual void setValue(VertexValue x) = 0;
    ALL virtual int numEdges() = 0;
    ALL virtual int numInEdges() = 0;
    ALL virtual int numOutEdges() = 0;

    ALL virtual ChiEdge<EdgeValue> *getInEdge(int i) = 0;
    ALL virtual ChiEdge<EdgeValue> *getOutEdge(int i) = 0;
    ALL virtual void setInEdge(int idx, int vertexId, EdgeValue value) = 0;
    ALL virtual void setOutEdge(VirtVertex<VertexValue, EdgeValue> **vertex,
                                int src, int idx, int vertexId,
                                EdgeValue value) = 0;
    ALL virtual ChiEdge<EdgeValue> *edge(int i) = 0;
    void set_in_out(obj_alloc *alloc) {
        this->inEdgeDataArray =
            (ChiEdge<EdgeValue> **)alloc->calloc<Edge<EdgeValue> *>(
                this->nInedges);
        this->outEdgeDataArray =
            (ChiEdge<EdgeValue> **)alloc->calloc<Edge<EdgeValue> *>(
                this->nOutedges);
        for (int i = 0; i < this->nInedges; i++) {
            this->inEdgeDataArray[i] =
                (Edge<EdgeValue> *)alloc->my_new<Edge<EdgeValue>>();
        }
    };
    ALL VirtVertex() {}
    ALL VirtVertex(int id, int inDegree, int outDegree, obj_alloc *alloc) {}
    ALL VirtVertex(int id, int inDegree, int outDegree) {}
    int id;
    int nInedges;
    int nOutedges;
    VertexValue value;
    ChiEdge<EdgeValue> **inEdgeDataArray;
    ChiEdge<EdgeValue> **outEdgeDataArray;
};

template <typename VertexValue, typename EdgeValue>
class ChiVertex : public VirtVertex<VertexValue, EdgeValue> {
  public:
    // init functions
    ALL ChiVertex() {}
    ChiVertex(int id, int inDegree, int outDegree, obj_alloc *alloc)
        : VirtVertex<VertexValue, EdgeValue>(id, inDegree, outDegree, alloc) {
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
            this->inEdgeDataArray[i] = alloc->my_new<Edge<EdgeValue>>();
        }
    }
    ALL ChiVertex(int id, int inDegree, int outDegree)
        : VirtVertex<VertexValue, EdgeValue>(id, inDegree, outDegree) {
        this->id = id;
        this->nInedges = inDegree;
        this->nOutedges = outDegree;
        // inEdgeDataArray =
        // (ChiEdge<EdgeValue>**)malloc(sizeof(ChiEdge<EdgeValue>*)*inDegree);
        // outEdgeDataArray =
        // (ChiEdge<EdgeValue>**)malloc(sizeof(ChiEdge<EdgeValue>*)*outDegree);
    }
    // operation functions
    ALL int getId() { return this->id; }
    ALL void setId(int x) { this->id = x; }
    ALL VertexValue getValue() { return this->value; }
    ALL void setValue(VertexValue x) { this->value = x; }
    ALL int numEdges() { return this->nInedges + this->nOutedges; }
    ALL int numInEdges() { return this->nInedges; }
    ALL int numOutEdges() { return this->nOutedges; }
    ALL ChiEdge<EdgeValue> *edge(int i) {
        if (i < this->nInedges)
            return this->inEdgeDataArray[i];
        else
            return this->inEdgeDataArray[i - this->nInedges];
    }
    ALL ChiEdge<EdgeValue> *getInEdge(int i) {
        return this->inEdgeDataArray[i];
    }
    ALL ChiEdge<EdgeValue> *getOutEdge(int i) {
        return this->outEdgeDataArray[i];
    }
    ALL void setInEdge(int idx, int vertexId, EdgeValue value) {
       new (CLEANPTR(this->inEdgeDataArray[idx],ChiEdge<EdgeValue> *))  Edge<EdgeValue>(vertexId, value);
        // inEdgeDataArray[idx] = new Edge<EdgeValue>(vertexId, value);
    }
    ALL void setOutEdge(VirtVertex<int, int> **vertex, int src, int idx,
                        int vertexId, EdgeValue value) {
        // if (print)
        //   printf("setOutEdge\n");
        // outEdgeDataArray[idx] = new Edge<EdgeValue>(vertexId, value);
        for (int i = 0; i < CLEANPTR(vertex[vertexId],VirtVertex<int COMMA int> *)->numInEdges(); i++) {
            ChiEdge<EdgeValue> * edge;
            edge = CLEANPTR(vertex[vertexId],VirtVertex<int COMMA int> *)->getInEdge(i);
            if (CLEANPTR (edge,ChiEdge<EdgeValue> *) ->getVertexId() == src) {
                this->outEdgeDataArray[idx] = CLEANPTR(vertex[vertexId],VirtVertex<int COMMA int> *)->getInEdge(i);
                break;
            }
        }
    }
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
