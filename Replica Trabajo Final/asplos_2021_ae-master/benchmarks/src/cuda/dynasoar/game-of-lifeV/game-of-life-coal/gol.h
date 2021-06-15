#ifndef EXAMPLE_GENERATION_SOA_GENERATION_H
#define EXAMPLE_GENERATION_SOA_GENERATION_H

#include "../configuration.h"
#include <new>

#include <stdio.h>
#include <assert.h>
#include "../../../mem_alloc/mem_alloc.h"

#include "coal.h"

__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size;
__managed__ void *temp_coal;




#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}
// Pre-declare all classes.
class Cell;
class Agent;
class Alive;
class Candidate;

enum AgentType { isAlive, isCandidate, None };
static const char kActionNone = 0;
static const char kActionDie = 1;
static const char kActionSpawnAlive = 2;

class AgentV {
public:
  int cell_id_;

protected:
  AgentType type;
  bool is_new_;

  int action_;

public:
  __device__ AgentV(int cell_id, AgentType type_) {}
  __device__ AgentV() {}
  __device__ virtual bool isAlive() = 0;
  __device__ virtual bool isCandidate() = 0;
  __device__ virtual bool is_new() = 0;
  __device__ virtual void set_is_new(bool is_new) = 0;



  __device__ virtual void set_action(int action) = 0;
  __device__ virtual int get_action() = 0;
  __device__ virtual int cell_id() = 0;
  __device__ virtual void update_checksum() = 0;
};

class Agent : public AgentV {

public:
  __device__ Agent(int cell_id, AgentType type_) {
    this->is_new_ = true;

    this->cell_id_ = (cell_id);
    this->action_ = (kActionNone);
    this->type = type_;
  }
  __device__ Agent() {
     this->type = AgentType::None;
  }
  
  __device__ bool isAlive() { return this->type == AgentType::isAlive; }
  __device__ bool isCandidate() { return this->type == AgentType::isCandidate; }
  __device__ bool is_new() { return is_new_; }
  __device__ void set_is_new(bool is_new) { is_new_ = is_new; }

  __device__ void set_action(int action) { this->action_ = action; }
  __device__ int get_action() { return this->action_; }
  __device__ int cell_id() { return this->cell_id_; }
#ifdef OPTION_RENDER
  // Only for rendering.
  __device__ void update_render_array();
#endif // OPTION_RENDER

  // Only for checksum computation.
  __device__ void update_checksum();
};

class CellV {
protected:
  AgentV *agent_;
  AgentV *private_agent;

public:
  int reserved;
  __device__ CellV() {}
  __host__ void inst_cell(obj_alloc *alloc){
     this->private_agent = (Agent *)alloc->my_new<Agent>();
  }
  __device__ virtual AgentV *agent() = 0;
  __device__ virtual void set_agent(int cid, AgentType type_) = 0;
  __device__ virtual void delete_agent() = 0;
  __device__ virtual bool is_empty() = 0;
};

class Cell : public CellV {

public:
  __device__ Cell() {

    //this->private_agent = new Agent();
    this->reserved = (0);
    this->agent_ = nullptr;
  }

  __device__ AgentV *agent() { return agent_; }
  __device__ void set_agent(int cid, AgentType type_) {
    this->agent_ = new (this->private_agent) Agent(cid, type_);
  }
  __device__ void delete_agent() { this->agent_ = nullptr; }
  __device__ bool is_empty() { return agent_ == nullptr; }
};

#endif // EXAMPLE_GENERATION_SOA_GENERATION_H
