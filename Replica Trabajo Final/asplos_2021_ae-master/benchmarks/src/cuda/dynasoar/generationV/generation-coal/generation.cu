#include <assert.h>
#include <stdio.h>
#include <chrono>

#include "../configuration.h"
#include "../dataset_loader.h"
#include "generation.h"
//#include "../rendering.h"
#include "coal.h"
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

// Dataset.
__device__ int SIZE_X;
__device__ int SIZE_Y;
__managed__ CellV **cells;
__managed__ Cell *cells2;
dataset_t dataset;
__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size;
__managed__ void *temp_coal;

// Only count alive agents in state 0.
__device__ int num_alive_neighbors(AgentV *ptr) {
    void **vtable;

    int cell_x = ptr->cell_id_ % SIZE_X;
    int cell_y = ptr->cell_id_ / SIZE_X;
    int result = 0;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                COAL_CellV_agent(cells[ny * SIZE_X + nx])
                AgentV *ptr = cells[ny * SIZE_X + nx]->agent();
                AgentV *alive = nullptr;
                if (ptr) {
                    COAL_AgentV_isAlive(ptr)
                    if (ptr->isAlive()) {
                        COAL_CellV_agent(cells[ny * SIZE_X + nx])
                        alive = cells[ny * SIZE_X + nx]->agent();
                    }
                }
                COAL_AgentV_is_state_equal(alive)
                if (alive != nullptr && alive->is_state_equal(0)) {
                    result++;
                }
            }
        }
    }

    return result;
}

__device__ void create_candidates(AgentV *ptr) {
    void **vtable;
    COAL_AgentV_is_new(ptr)
    assert(ptr->is_new());

    // TODO: Consolidate with Agent::num_alive_neighbors().
    int cell_x = ptr->cell_id_ % SIZE_X;
    int cell_y = ptr->cell_id_ / SIZE_X;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                auto cid = ny * SIZE_X + nx;
                COAL_CellV_is_empty(cells[cid])
                if (cells[cid]->is_empty()) {
                    if (atomicCAS(&cells[cid]->reserved, 0, 1) == 0) {
                        COAL_CellV_set_agent(cells[cid])
                        cells[cid]->set_agent(cid, AgentType::isCandidate);
                    }
                }
            }
        }
    }
}

__global__ void create_cells() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        cells[i] = new (&cells2[i]) Cell();
        assert(cells[i] != nullptr);
    }
}

// Must be followed by Agent::update().
__global__ void load_game(int *cell_ids, int num_cells) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_cells;
         i += blockDim.x * gridDim.x) {
        cells[cell_ids[i]]->set_agent(cell_ids[i], AgentType::isAlive);
        assert(cells[cell_ids[i]]->agent() != nullptr);
        assert(cells[cell_ids[i]]->agent()->cell_id() == cell_ids[i]);
    }
}

__device__ void alive_update_2(int i) {
    AgentV *ptr = cells[i]->agent();
    if (ptr)
        if (ptr->isAlive()) {
            int cid = ptr->cell_id_;

            // TODO: Consider splitting in two classes for less divergence.
            if (ptr->is_new()) {
                // Create candidates in neighborhood.
                create_candidates(ptr);
            } else {
                if (ptr->get_action() == kActionDie && ptr->is_state_equal(0)) {
                    // Increment state. If reached max. state, replace with
                    // Candidate.
                    ptr->inc_state();
                    ptr->set_action(kActionNone);
                } else if (ptr->is_state_in_range(0, kNumStates)) {
                    ptr->inc_state();
                } else if (ptr->is_state_equal(kNumStates)) {
                    // Replace with Candidate.
                    cells[cid]->set_agent(cid, AgentType::isCandidate);
                    // delete this;
                }
            }
        }
}

__global__ void prepare() {
    void **vtable;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        COAL_CellV_agent(cells[i])
        AgentV *ptr = cells[i]->agent();
        if (ptr) {
           COAL_AgentV_isAlive(ptr)
            if (ptr->isAlive()) {
                COAL_AgentV_is_state_equal(ptr)
                if (ptr->is_state_equal(0)) {
                    COAL_AgentV_set_is_new(ptr)
                    ptr->set_is_new(false);

                    // Also counts this object itself.
                    int alive_neighbors = num_alive_neighbors(ptr) - 1;

                    const bool stay_alive_param[9] = kStayAlive;
                    if (!stay_alive_param[alive_neighbors]) {
                        COAL_AgentV_set_action(ptr)
                        ptr->set_action(kActionDie);
                    }
                }
            }
        }
        if (ptr) {
            COAL_AgentV_isCandidate(ptr)
            if (ptr->isCandidate()) {
                int alive_neighbors = num_alive_neighbors(ptr);
                const bool spawn_param[9] = kSpawnNew;

                if (spawn_param[alive_neighbors]) {
                    COAL_AgentV_set_action(ptr)
                    ptr->set_action(kActionSpawnAlive);
                } else if (alive_neighbors == 0) {
                    COAL_AgentV_set_action(ptr)

                    ptr->set_action(kActionDie);
                }
            }
        }
    }
}
__global__ void update() {
    void **vtable;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        COAL_CellV_agent(cells[i])
        AgentV *ptr = cells[i]->agent();
        if (ptr) {
            COAL_AgentV_isCandidate(ptr)
            if (ptr->isCandidate()) {
                int cid = ptr->cell_id_;
                COAL_AgentV_get_action(ptr)
                if (ptr->get_action() == kActionSpawnAlive) {
                    COAL_CellV_set_agent(cells[cid])
                    cells[cid]->set_agent(cid, AgentType::isAlive);
                    // delete this;
                } else if (ptr->get_action() == kActionDie) {
                    COAL_CellV_delete_agent(cells[cid])
                    cells[cid]->delete_agent();
                    cells[cid]->reserved = 0;
                    // delete this;
                }
            }
        }
        // alive_update_2(i);
    }
}
__global__ void alive_update() {
    void **vtable;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        COAL_CellV_agent(cells[i])
        AgentV *ptr = cells[i]->agent();
        if (ptr) {
            COAL_AgentV_isAlive(ptr)
            if (ptr->isAlive()) {
                int cid = ptr->cell_id_;
                COAL_AgentV_is_new(ptr)
                // TODO: Consider splitting in two classes for less divergence.
                if (ptr->is_new()) {
                    // Create candidates in neighborhood.
                    create_candidates(ptr);
                } else {
                    COAL_AgentV_get_action(ptr)
                    bool flag1 = ptr->get_action() == kActionDie;

                    COAL_AgentV_is_state_equal(ptr)
                    flag1 = flag1 && ptr->is_state_equal(0);

                    COAL_AgentV_is_state_in_range(ptr)
                    bool flag2 = ptr->is_state_in_range(0, kNumStates);
                    COAL_AgentV_is_state_equal(ptr)
                    bool flag3 = ptr->is_state_equal(kNumStates);
                    if (flag1) {
                        // Increment state. If reached max. state, replace with
                        // Candidate.
                        COAL_AgentV_inc_state(ptr)
                        ptr->inc_state();
                        COAL_AgentV_set_action(ptr)
                        ptr->set_action(kActionNone);
                    } else if (flag2) {
                        COAL_AgentV_inc_state(ptr)
                        ptr->inc_state();
                    } else if (flag3) {
                        // Replace with Candidate.
                        COAL_CellV_set_agent(cells[cid])
                        cells[cid]->set_agent(cid, AgentType::isCandidate);
                        // delete this;
                    }
                }
            }
        }
    }
}

__global__ void update_checksum() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        AgentV *ptr = cells[i]->agent();
        if (ptr) ptr->update_checksum();
    }
}

void transfer_dataset() {
    int *dev_cell_ids;
    int num_alive = dataset.alive_cells.size();
    printf("number of alive %d \n", num_alive);
    cudaMalloc(&dev_cell_ids, sizeof(int) * num_alive);
    cudaMemcpy(dev_cell_ids, dataset.alive_cells.data(),
               sizeof(int) * num_alive, cudaMemcpyHostToDevice);

#ifndef NDEBUG
    printf("Loading on GPU: %i alive cells.\n", num_alive);
#endif  // NDEBUG

    load_game<<<1024, 1024>>>(dev_cell_ids, num_alive);
    gpuErrchk(cudaDeviceSynchronize());
    // cudaFree(dev_cell_ids);

    alive_update<<<1024, 1024>>>();
    gpuErrchk(cudaDeviceSynchronize());
}

__device__ int device_checksum;
__device__ int device_num_candidates;

__device__ __noinline__ void Agent::update_checksum() {
    if (this->isAlive())
        atomicAdd(&device_checksum, 1);
    else
        atomicAdd(&device_num_candidates, 1);
}

int checksum() {
    int host_checksum = 0;
    int host_num_candidates = 0;
    cudaMemcpyToSymbol(device_checksum, &host_checksum, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device_num_candidates, &host_num_candidates, sizeof(int),
                       0, cudaMemcpyHostToDevice);

    update_checksum<<<1024, 1024>>>();
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpyFromSymbol(&host_checksum, device_checksum, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_num_candidates, device_num_candidates,
                         sizeof(int), 0, cudaMemcpyDeviceToHost);

    return host_checksum + host_num_candidates;
}

int main(int argc, char ** argv) {
    // Load data set.
    dataset = load_burst();

    cudaMemcpyToSymbol(SIZE_X, &dataset.x, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(SIZE_Y, &dataset.y, sizeof(int), 0,
                       cudaMemcpyHostToDevice);

#ifdef OPTION_RENDER
    init_renderer();
#endif  // OPTION_RENDER

    // Allocate memory.

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);
    mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
    obj_alloc my_obj_alloc(&shared_mem, atoll(argv[1]));
    // cudaMalloc(&cells, sizeof(Cell *) * dataset.x * dataset.y);
    // cudaMalloc(&cells2, sizeof(Cell) * dataset.x * dataset.y);
    cells = (CellV **)my_obj_alloc.calloc<CellV *>(dataset.x * dataset.y);
    for (int i = 0; i < dataset.x * dataset.y; i++) {
        cells[i] = (Cell *)my_obj_alloc.my_new<Cell>();
        cells[i]->inst_cell(&my_obj_alloc);
        // assert(cells[i] != nullptr);
    }
    my_obj_alloc.toDevice();
    my_obj_alloc.create_tree();
    range_tree = my_obj_alloc.get_range_tree();
    tree_size = my_obj_alloc.get_tree_size();
    // Initialize cells.
    // create_cells<<<1024, 1024>>>();
    gpuErrchk(cudaDeviceSynchronize());

    transfer_dataset();

    auto time_start = std::chrono::system_clock::now();

    // Run simulation.
    for (int i = 0; i < kNumIterations; ++i) {
#ifdef OPTION_RENDER
        render();
#endif  // OPTION_RENDER

#ifndef NDEBUG
        if (i % 30 == 0) printf("%i\n", i);

#endif  // NDEBUG

        // can_prepare<<<1024, 1024>>>();
        // gpuErrchk(cudaDeviceSynchronize());
        prepare<<<1024, 1024>>>();
        gpuErrchk(cudaDeviceSynchronize());
        update<<<1024, 1024>>>();
        gpuErrchk(cudaDeviceSynchronize());
        alive_update<<<1024, 1024>>>();
        gpuErrchk(cudaDeviceSynchronize());
        // alive_update<<<1024, 1024>>>();
        // gpuErrchk(cudaDeviceSynchronize());
    }

    auto time_end = std::chrono::system_clock::now();
    auto elapsed = time_end - time_start;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

#ifdef OPTION_RENDER
    close_renderer();
#endif  // OPTION_RENDER

#ifndef NDEBUG
    printf("Checksum: %i \n", checksum());
#endif  // NDEBUG

    printf("%lu, \n", micros);

    //  if (kOptionPrintStats) {
    //    allocator_handle->DBG_print_collected_stats();
    //  }

#ifdef OPTION_RENDER
    delete[] host_render_cells;
    cudaFree(d_device_render_cells);
#endif  // OPTION_RENDER

    return 0;
}
