#define ALL __noinline__ __host__ __device__
template <typename EdgeData> class ChiEdge {
public:
  int type;
  ALL int getVertexIdC() {return -1;}
  ALL EdgeData getValueC() {return 0;}
  ALL void setValueC(EdgeData x) {}
  ALL int getVertexIdV() { return this->vertexId; }
  ALL EdgeData getValueV() { return this->edgeValue; }
  ALL void setValueV(EdgeData x) { this->edgeValue = x; }

  EdgeData edgeValue;
  int vertexId;
};

template <typename EdgeValue> class Edge : public ChiEdge<EdgeValue> {
public:
  ALL Edge(int id, int value) {
    this->vertexId = id;
    this->edgeValue = value;
    this->type = 1;
  }

};
template <typename VertexValue, typename EdgeValue>
class ChiVertex;

template <typename VertexValue, typename EdgeValue>
class VirtVertex {
    public:
	// operation functions
	ALL int getIdC()  { return 0;}
	ALL void setIdC(int x)  { return; }
	ALL VertexValue getValueC()  { return 0;}
	ALL void setValueC(VertexValue x) { return; }
	ALL int numInEdgesC() { return 0;}
	ALL int numOutEdgesC() { return 0;}
	ALL ChiEdge<EdgeValue>* getInEdgeC(int i) { return 0;}
	ALL ChiEdge<EdgeValue>* getOutEdgeC(int i){ return 0;}
	ALL void setInEdgeC(int idx, int vertexId, EdgeValue value){ return; }
	ALL void setOutEdgeC(VirtVertex<int, int> **vertex, int src, int idx, int vertexId, EdgeValue value) { return; }


	ALL int getIdV() {
	    return id;
	}
	ALL void setIdV(int x) {
	    id = x;
	}
	ALL VertexValue getValueV() {
	    return value;
	}
	ALL void setValueV(VertexValue x) {
	    value = x;
	}
	ALL int numInEdgesV() {
	    return nInedges;
	}
	ALL int numOutEdgesV() {
	    return nOutedges;
	}
	ALL ChiEdge<EdgeValue>* getInEdgeV(int i) {
	    return inEdgeDataArray[i];
	}
	ALL ChiEdge<EdgeValue>* getOutEdgeV(int i) {
	    return outEdgeDataArray[i];
	}
	ALL void setInEdgeV(int idx, int vertexId, EdgeValue value) {
	    inEdgeDataArray[idx] = new Edge<EdgeValue>(vertexId, value);
	}
	ALL void setOutEdgeV(VirtVertex<int, int> **vertex, int src, int idx, int vertexId, EdgeValue value) {
	    //outEdgeDataArray[idx] = new Edge<EdgeValue>(vertexId, value);
	    for (int i = 0; i < vertex[vertexId]->numInEdgesV(); i++) {
	        if (vertex[vertexId]->getInEdgeV(i)->getVertexIdV() == src) {
	            outEdgeDataArray[idx] = vertex[vertexId]->getInEdgeV(i);
	            break;
	        }
	    }
	}
    public:
	int type;
	int id;
	int nInedges;
	int nOutedges;
	VertexValue value;
	ChiEdge<EdgeValue>** inEdgeDataArray;
	ChiEdge<EdgeValue>** outEdgeDataArray;
};

template <typename VertexValue, typename EdgeValue>
class ChiVertex : public VirtVertex<VertexValue, EdgeValue> {
    public:
	// init functions
	ALL ChiVertex(int id, int inDegree, int outDegree) {
	    this->id = id;
		this->type=1;
	    this->nInedges =  inDegree;
	   	this->nOutedges = outDegree;
	  	this->  inEdgeDataArray = (ChiEdge<EdgeValue>**)malloc(sizeof(ChiEdge<EdgeValue>*)*inDegree);
	   	this-> outEdgeDataArray = (ChiEdge<EdgeValue>**)malloc(sizeof(ChiEdge<EdgeValue>*)*outDegree);
	}
	// operation functions
	

};

class GraphChiContext  {
    public:
	ALL GraphChiContext(int vert) {
	}
	ALL int getNumIterations() {
	    return numIterations;
	}

	ALL void setNumIterations(int iter) {
	    numIterations = iter;
	}
	ALL int getNumVertices() {
	    return numVertices;
	}

	ALL void setNumVertices(int vertices) {
	    numVertices = vertices;
	}
	ALL int getNumEdges() {
	    return numEdges;
	}

	ALL void setNumEdges(int edges) {
	    numEdges = edges;
	}
    private:
	int numIterations;
	int numVertices;
	int numEdges;
};
