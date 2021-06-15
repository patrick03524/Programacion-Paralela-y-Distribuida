#define COAL_AgentV_isAlive(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_AgentV_isCandidate(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_AgentV_is_new(ptr)                         \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_AgentV_set_is_new(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
#define COAL_AgentV_set_action(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[4];                          \
    }
#define COAL_AgentV_get_action(ptr)                     \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[5];                          \
    }
#define COAL_AgentV_cell_id(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[6];                          \
    }
#define COAL_AgentV_update_checksum(ptr)                \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[7];                          \
    }
#define COAL_CellV_agent(ptr)                           \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[0];                          \
    }
#define COAL_CellV_set_agent(ptr)                       \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[1];                          \
    }
#define COAL_CellV_delete_agent(ptr)                    \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[2];                          \
    }
#define COAL_CellV_is_empty(ptr)                        \
    {                                                   \
        vtable = get_vfunc(ptr, range_tree, tree_size); \
        temp_coal = vtable[3];                          \
    }
