#define COAL_AgentV_isAlive(ptr)                        \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[0];                          \
    }
#define COAL_AgentV_isCandidate(ptr)                    \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[1];                          \
    }
#define COAL_AgentV_is_new(ptr)                         \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[2];                          \
    }
#define COAL_AgentV_set_is_new(ptr)                     \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[3];                          \
    }
#define COAL_AgentV_is_state_equal(ptr)                 \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[4];                          \
    }
#define COAL_AgentV_set_state(ptr)                      \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[5];                          \
    }
#define COAL_AgentV_inc_state(ptr)                      \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[6];                          \
    }
#define COAL_AgentV_is_state_in_range(ptr)              \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[7];                          \
    }
#define COAL_AgentV_set_action(ptr)                     \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[8];                          \
    }
#define COAL_AgentV_get_action(ptr)                     \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[9];                          \
    }
#define COAL_AgentV_cell_id(ptr)                        \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[10];                         \
    }
#define COAL_AgentV_update_checksum(ptr)                \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[11];                         \
    }
#define COAL_CellV_agent(ptr)                           \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[0];                          \
    }
#define COAL_CellV_set_agent(ptr)                       \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[1];                          \
    }
#define COAL_CellV_delete_agent(ptr)                    \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[2];                          \
    }
#define COAL_CellV_is_empty(ptr)                        \
    {                                                   \
        vtable = get_vfunc_type(ptr, vfun_table); \
        temp_TP = vtable[3];                          \
    }
