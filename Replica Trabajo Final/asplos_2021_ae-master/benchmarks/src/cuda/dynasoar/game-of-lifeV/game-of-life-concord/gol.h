#ifndef EXAMPLE_GENERATION_SOA_GENERATION_H
#define EXAMPLE_GENERATION_SOA_GENERATION_H

#include <new>
#include "../configuration.h"

#include <assert.h>
#include <stdio.h>


#define CONCORD2(ptr, fun)        \
    if (ptr->classType == 0)      \
        ptr->Base##fun;           \
    else if (ptr->classType == 1) \
        ptr->fun;
#define CONCORD3(r, ptr, fun)     \
    if (ptr->classType == 0)      \
        r = ptr->Base##fun;       \
    else if (ptr->classType == 1) \
        r = ptr->fun;

#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define CONCORD(...) \
    GET_MACRO(__VA_ARGS__, CONCORD4, CONCORD3, CONCORD2, CONCORD1)(__VA_ARGS__)

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
    int classType = 0;

  protected:
    AgentType type;
    bool is_new_;

    int action_;

  public:
    __device__ __host__ __noinline__ AgentV(int cell_id, AgentType type_) {
        classType = 0;
    }
    __device__ __host__ __noinline__ AgentV() { classType = 0; }
     __device__ __host__ __noinline__ bool BaseisAlive() {
        return this->type == AgentType::isAlive;
    }
    __device__ __host__ __noinline__ bool BaseisCandidate() {
        return this->type == AgentType::isCandidate;
    }
    __device__ __host__ __noinline__ bool Baseis_new() { return is_new_; }
    __device__ __host__ __noinline__ void Baseset_is_new(bool is_new) {
        is_new_ = is_new;
    }

    __device__ __host__ __noinline__ void Baseset_action(int action) {
        this->action_ = action;
    }
    __device__ __host__ __noinline__ int Baseget_action() { return this->action_; }
    __device__ __host__ __noinline__ int Basecell_id() { return this->cell_id_; }
#ifdef OPTION_RENDER
    // Only for rendering.
    __device__ __host__ __noinline__ void update_render_array();
#endif  // OPTION_RENDER

    // Only for checksum computation.
    __device__ __noinline__ void Baseupdate_checksum(){}
   
    __device__ __host__ __noinline__ bool isAlive() {
        return this->type == AgentType::isAlive;
    }
    __device__ __host__ __noinline__ bool isCandidate() {
        return this->type == AgentType::isCandidate;
    }
    __device__ __host__ __noinline__ bool is_new() { return is_new_; }
    __device__ __host__ __noinline__ void set_is_new(bool is_new) {
        is_new_ = is_new;
    }

    __device__ __host__ __noinline__ void set_action(int action) {
        this->action_ = action;
    }
    __device__ __host__ __noinline__ int get_action() { return this->action_; }
    __device__ __host__ __noinline__ int cell_id() { return this->cell_id_; }
#ifdef OPTION_RENDER
    // Only for rendering.
    __device__ __host__ __noinline__ void update_render_array();
#endif  // OPTION_RENDER

    // Only for checksum computation.
    __device__ __noinline__ void update_checksum();
};

class Agent : public AgentV {
  public:
    __device__ __host__ __noinline__ Agent(int cell_id, AgentType type_) {
        this->is_new_ = true;
        this->cell_id_ = (cell_id);
        this->action_ = (kActionNone);
        this->type = type_;
        classType = 1;
    }
    __device__ __host__ __noinline__ Agent() {
        this->type = AgentType::None;
        classType = 1;
    }
};

class CellV {
  protected:
    AgentV *agent_;
    AgentV *private_agent;
   
  public:
    int reserved;
     int classType =0; 

    __device__ __host__ __noinline__ CellV() {classType =0; }



    __device__ __host__ __noinline__ AgentV *Baseagent() { return agent_; }
    __device__ __host__ __noinline__ void Baseset_agent(int cid, AgentType type_) {
        this->agent_ = new (this->private_agent) Agent(cid, type_);
    }
    __device__ __host__ __noinline__ void Basedelete_agent() {
        this->agent_ = nullptr;
    }
    __device__ __host__ __noinline__ bool Baseis_empty() {
        return agent_ == nullptr;
    }
    __device__ __host__ __noinline__ AgentV *agent() { return agent_; }
    __device__ __host__ __noinline__ void set_agent(int cid, AgentType type_) {
        this->agent_ = new (this->private_agent) Agent(cid, type_);
    }
    __device__ __host__ __noinline__ void delete_agent() {
        this->agent_ = nullptr;
    }
    __device__ __host__ __noinline__ bool is_empty() {
        return agent_ == nullptr;
    }
};

class Cell : public CellV {
  public:
    __device__ __host__ __noinline__ Cell() {
        this->private_agent = new Agent();
        this->reserved = (0);
        this->agent_ = nullptr;
        classType =1; 
    }
};

#endif  // EXAMPLE_GENERATION_SOA_GENERATION_H
