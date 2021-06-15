#ifndef EXAMPLE_GENERATION_SOA_GENERATION_H
#define EXAMPLE_GENERATION_SOA_GENERATION_H

#include "../configuration.h"

#define ALL __noinline__ __host__ __device__
#define CONCORD2(ptr,fun) \
if (ptr->classType==0) ptr->Base##fun  ; else if (ptr->classType==1) ptr->fun  ;
#define CONCORD3(r,ptr,fun) \
if (ptr->classType==0) r  = ptr->Base##fun  ; else if (ptr->classType==1) r = ptr->fun  ;

#define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME
#define CONCORD(...) GET_MACRO(__VA_ARGS__, CONCORD4, CONCORD3,CONCORD2,CONCORD1)(__VA_ARGS__)

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
    int classType = 0;

  protected:
    AgentType type;
    bool is_new_;
    int state_;
    int action_;

  public:
    ALL AgentV(int cell_id, AgentType type_) { classType = 0; }
    ALL AgentV() { classType = 0; }

    __device__ __noinline__ void Baseupdate_checksum() {}
    ////////////////
    ALL bool BaseisAlive() { return this->type == AgentType::isAlive; }
    ALL bool BaseisCandidate() { return this->type == AgentType::isCandidate; }
    ALL bool Baseis_new() { return is_new_; }
    ALL void Baseset_is_new(bool is_new) { is_new_ = is_new; }
    ALL bool Baseis_state_equal(int state) { return this->state_ == state; }
    ALL void Baseset_state(int state) { this->state_ = state; }
    ALL void Baseinc_state() { this->state_++; }
    ALL bool Baseis_state_in_range(int min, int max) {
        return this->state_ > min && this->state_ < max;
    }
    ALL void Baseset_action(int action) { this->action_ = action; }
    ALL int   Baseget_action() { return this->action_; }
    ALL int Basecell_id() { return this->cell_id_; }
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
#endif  // OPTION_RENDER

    // Only for checksum computation.
    __device__ __noinline__ void update_checksum();
};

class Agent : public AgentV {
  public:
    ALL Agent(int cell_id, AgentType type_) {
        this->is_new_ = true;
        this->state_ = 0;
        this->cell_id_ = (cell_id);
        this->action_ = (kActionNone);
        this->type = type_;
        classType = 1;
    }
    ALL Agent() { classType = 1; }
};

class CellV {
  protected:
    AgentV *agent_;
    AgentV *private_agent;

  public:
    int reserved;
    int classType = 0;
    ALL CellV() { classType = 0; }

    ALL AgentV *Baseagent() { return agent_; }
    ALL void Baseset_agent(int cid, AgentType type_) {
        this->agent_ = new (this->private_agent) Agent(cid, type_);
    }
    ALL void Basedelete_agent() { this->agent_ = nullptr; }
    ALL bool Baseis_empty() { return agent_ == nullptr; }

    ALL AgentV *agent() { return agent_; }
    ALL void set_agent(int cid, AgentType type_) {
        this->agent_ = new (this->private_agent) Agent(cid, type_);
    }
    ALL void delete_agent() { this->agent_ = nullptr; }
    ALL bool is_empty() { return agent_ == nullptr; }
};

class Cell : public CellV {
  public:
    ALL Cell() {
        this->private_agent = new Agent();
        this->reserved = (0);
        this->agent_ = nullptr;
        classType = 1;
    }
};

#endif  // EXAMPLE_GENERATION_SOA_GENERATION_H
