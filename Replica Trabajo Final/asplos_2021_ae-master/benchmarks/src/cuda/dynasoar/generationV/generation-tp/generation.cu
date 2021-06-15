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

dataset_t dataset;
__managed__ obj_info_tuble *vfun_table;
__managed__ unsigned tree_size_g;
__managed__ void *temp_copyBack;
__managed__ void *temp_TP;


// Only count alive agents in state 0.
__device__ int num_alive_neighbors(AgentV *ptr) {
    void **vtable;

    int cell_x = CLEANPTR( ptr ,AgentV *)->cell_id_ % SIZE_X;
    int cell_y = CLEANPTR( ptr ,AgentV *)->cell_id_ / SIZE_X;
    int result = 0;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                COAL_CellV_agent(cells[ny * SIZE_X + nx])
                AgentV *ptr = CLEANPTR( cells[ny * SIZE_X + nx] ,CellV *)->agent();
                AgentV *alive = nullptr;
                if (ptr) {
                    COAL_AgentV_isAlive(ptr)
                    if (CLEANPTR( ptr ,AgentV *)->isAlive()) {
                        COAL_CellV_agent(cells[ny * SIZE_X + nx])
                        alive = CLEANPTR( cells[ny * SIZE_X + nx] ,CellV *)->agent();
                    }
                }
                COAL_AgentV_is_state_equal(alive)
                if (alive != nullptr && CLEANPTR( alive ,AgentV * )->is_state_equal(0)) {
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
    assert(CLEANPTR( ptr ,AgentV *)->is_new());

    // TODO: Consolidate with Agent::num_alive_neighbors().
    int cell_x = CLEANPTR( ptr ,AgentV *)->cell_id_ % SIZE_X;
    int cell_y = CLEANPTR( ptr ,AgentV *)->cell_id_ / SIZE_X;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                auto cid = ny * SIZE_X + nx;
                COAL_CellV_is_empty(cells[cid])
                if (CLEANPTR( cells[cid] ,CellV *)->is_empty()) {
                    if (atomicCAS(&CLEANPTR( cells[cid] ,CellV *)->reserved, 0, 1) == 0) {
                        COAL_CellV_set_agent(cells[cid])
                        CLEANPTR( cells[cid] ,CellV *)->set_agent(cid, AgentType::isCandidate);
                    }
                }
            }
        }
    }
}



// Must be followed by Agent::update().
__global__ void load_game(int *cell_ids, int num_cells) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_cells;
         i += blockDim.x * gridDim.x) {
        CLEANPTR( cells[cell_ids[i]] ,CellV *)->set_agent(cell_ids[i], AgentType::isAlive);
        assert(CLEANPTR( cells[cell_ids[i]] ,CellV *)->agent() != nullptr);
        assert(CLEANPTR(CLEANPTR( cells[cell_ids[i]] ,CellV *)->agent(),AgentV *)->cell_id() == cell_ids[i]);
    }
}



__global__ void prepare() {
    void **vtable;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        COAL_CellV_agent(cells[i])
        AgentV *ptr = CLEANPTR( cells[i] ,CellV *)->agent();
        if (ptr) {
           COAL_AgentV_isAlive(ptr)
            if (CLEANPTR( ptr ,AgentV *)->isAlive()) {
                COAL_AgentV_is_state_equal(ptr)
                if (CLEANPTR( ptr ,AgentV *)->is_state_equal(0)) {
                    COAL_AgentV_set_is_new(ptr)
                    CLEANPTR( ptr ,AgentV *)->set_is_new(false);

                    // Also counts this object itself.
                    int alive_neighbors = num_alive_neighbors(ptr) - 1;

                    const bool stay_alive_param[9] = kStayAlive;
                    if (!stay_alive_param[alive_neighbors]) {
                        COAL_AgentV_set_action(ptr)
                        CLEANPTR( ptr ,AgentV *)->set_action(kActionDie);
                    }
                }
            }
        }
        if (ptr) {
            COAL_AgentV_isCandidate(ptr)
            if (CLEANPTR( ptr ,AgentV *)->isCandidate()) {
                int alive_neighbors = num_alive_neighbors(ptr);
                const bool spawn_param[9] = kSpawnNew;

                if (spawn_param[alive_neighbors]) {
                    COAL_AgentV_set_action(ptr)
                    CLEANPTR( ptr ,AgentV *)->set_action(kActionSpawnAlive);
                } else if (alive_neighbors == 0) {
                    COAL_AgentV_set_action(ptr)

                    CLEANPTR( ptr ,AgentV *)->set_action(kActionDie);
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
        AgentV *ptr = CLEANPTR( cells[i] ,CellV * )->agent();
        if (ptr) {
            COAL_AgentV_isCandidate(ptr)
            if (CLEANPTR( ptr ,AgentV *)->isCandidate()) {
                int cid = CLEANPTR( ptr ,AgentV *)->cell_id_;
                COAL_AgentV_get_action(ptr)
                if (CLEANPTR( ptr ,AgentV *)->get_action() == kActionSpawnAlive) {
                    COAL_CellV_set_agent(cells[cid])
                    CLEANPTR( cells[cid] ,CellV *)->set_agent(cid, AgentType::isAlive);
                    // delete this;
                } else if (CLEANPTR( ptr ,AgentV *)->get_action() == kActionDie) {
                    COAL_CellV_delete_agent(cells[cid])
                    CLEANPTR( cells[cid] ,CellV *)->delete_agent();
                    CLEANPTR( cells[cid] ,CellV *)->reserved = 0;
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
        AgentV *ptr = CLEANPTR( cells[i] ,CellV *)->agent();
        // printf("%p\n",ptr);
        if (CLEANPTR( ptr ,AgentV * )) {
            COAL_AgentV_isAlive(ptr)
            if (CLEANPTR( ptr ,AgentV * )->isAlive()) {
                int cid = CLEANPTR( ptr ,AgentV * )->cell_id_;
                COAL_AgentV_is_new(ptr)
                // TODO: Consider splitting in two classes for less divergence.
                if (CLEANPTR( ptr ,AgentV * )->is_new()) {
                    // Create candidates in neighborhood.
                    create_candidates(ptr);
                } else {
                    COAL_AgentV_get_action(ptr)
                    bool flag1 = CLEANPTR( ptr ,AgentV * )->get_action() == kActionDie;

                    COAL_AgentV_is_state_equal(ptr)
                    flag1 = flag1 && CLEANPTR( ptr ,AgentV * )->is_state_equal(0);

                    COAL_AgentV_is_state_in_range(ptr)
                    bool flag2 = CLEANPTR( ptr ,AgentV * )->is_state_in_range(0, kNumStates);
                    COAL_AgentV_is_state_equal(ptr)
                    bool flag3 = CLEANPTR( ptr ,AgentV * )->is_state_equal(kNumStates);
                    if (flag1) {
                        // Increment state. If reached max. state, replace with
                        // Candidate.
                        COAL_AgentV_inc_state(ptr)
                        CLEANPTR( ptr ,AgentV * )->inc_state();
                        COAL_AgentV_set_action(ptr)
                        CLEANPTR( ptr ,AgentV * )->set_action(kActionNone);
                    } else if (flag2) {
                        COAL_AgentV_inc_state(ptr)
                        CLEANPTR( ptr ,AgentV * )->inc_state();
                    } else if (flag3) {
                        // Replace with Candidate.
                        COAL_CellV_set_agent(cells[cid])
                        CLEANPTR( cells[cid] ,CellV * )->set_agent(cid, AgentType::isCandidate);
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
        AgentV *ptr = CLEANPTR( cells[i] ,CellV *)->agent();
        if (CLEANPTR( ptr ,AgentV * )) CLEANPTR( ptr ,AgentV * )->update_checksum();
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

    cells = (CellV **)my_obj_alloc.calloc<CellV *>(dataset.x * dataset.y);
    for (int i = 0; i < dataset.x * dataset.y; i++) {
        cells[i] = (Cell *)my_obj_alloc.my_new<Cell>();
        CLEANPTR( cells[i] ,CellV *)->inst_cell(&my_obj_alloc);
        // assert(cells[i] != nullptr);
    }
    my_obj_alloc.toDevice();
    my_obj_alloc.create_table();
    vfun_table = my_obj_alloc.get_vfun_table();
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
