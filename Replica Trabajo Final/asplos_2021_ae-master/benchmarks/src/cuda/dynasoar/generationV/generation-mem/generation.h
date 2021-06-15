#ifndef EXAMPLE_GENERATION_SOA_GENERATION_H
#define EXAMPLE_GENERATION_SOA_GENERATION_H

#include "../configuration.h"
#include "../../../mem_alloc/mem_alloc.h"
#define ALL __noinline__ __host__ __device__
// Pre-declare all classes.
class Cell;
class Agent;
class Alive;
class Candidate;

enum AgentType { isAlive, isCandidate };
static const char kActionNone = 0;
static const char kActionDie = 1;
static const char kActionSpawnAlive = 2;

class AgentV {
public:
  int cell_id_;

protected:
  AgentType type;
  bool is_new_;
  int state_;
  int action_;

public:
  ALL AgentV(int cell_id, AgentType type_) {}
  ALL AgentV() {}
  ALL virtual bool isAlive() = 0;
  ALL virtual bool isCandidate() = 0;
  ALL virtual bool is_new() = 0;
  ALL virtual void set_is_new(bool is_new) = 0;
  ALL virtual bool is_state_equal(int state) = 0;
  ALL virtual void set_state(int state) = 0;
  ALL virtual void inc_state() = 0;
  ALL virtual bool is_state_in_range(int min, int max) = 0;
  ALL virtual void set_action(int action) = 0;
  ALL virtual int get_action() = 0;
  ALL virtual int cell_id() = 0;
   __device__ __noinline__ virtual void update_checksum() = 0;
};

class Agent : public AgentV {

public:
  ALL Agent(int cell_id, AgentType type_) {
    this->is_new_ = true;
    this->state_ = 0;
    this->cell_id_ = (cell_id);
    this->action_ = (kActionNone);
    this->type = type_;
  }
  ALL __host__ Agent() {}

  ALL bool isAlive() { return this->type == AgentType::isAlive; }
  ALL bool isCandidate() { return this->type == AgentType::isCandidate; }
  ALL bool is_new() { return is_new_; }
  ALL void set_is_new(bool is_new) { is_new_ = is_new; }

  ALL bool is_state_equal(int state) { return this->state_ == state; }
  ALL void set_state(int state) { this->state_ = state; }
  ALL void inc_state() { this->state_++; }
  ALL bool is_state_in_range(int min, int max) {
    return this->state_ > min && this->state_ < max;
  }
  ALL void set_action(int action) { this->action_ = action; }
  ALL int get_action() { return this->action_; }
  ALL int cell_id() { return this->cell_id_; }
#ifdef OPTION_RENDER
  // Only for rendering.
  ALL void update_render_array();
#endif // OPTION_RENDER

  // Only for checksum computation.
   __device__ __noinline__ void update_checksum();
};

class CellV {
protected:
  AgentV *agent_;
  AgentV *private_agent;

public:
  int reserved;
  ALL CellV() {}
  __host__ void inst_cell(obj_alloc *alloc){
     this->private_agent = (Agent *)alloc->my_new<Agent>();
  }
  ALL virtual AgentV *agent() = 0;
  ALL virtual void set_agent(int cid, AgentType type_) = 0;
  ALL virtual void delete_agent() = 0;
  ALL virtual bool is_empty() = 0;
};

class Cell : public CellV {

public:
  ALL __host__ Cell() {

   
    this->reserved = (0);
    this->agent_ = nullptr;
  }
 

  ALL AgentV *agent() { return agent_; }
  ALL void set_agent(int cid, AgentType type_) {
    this->agent_ = new (this->private_agent) Agent(cid, type_);
  }
  ALL void delete_agent() { this->agent_ = nullptr; }
  ALL bool is_empty() { return agent_ == nullptr; }
};

#endif // EXAMPLE_GENERATION_SOA_GENERATION_H
