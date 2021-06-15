#include <assert.h>
#include <stdio.h>
#include <chrono>

#include "../configuration.h"
#include "../dataset_loader.h"
#include "generation.h"
//#include "../rendering.h"

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
#ifdef OPTION_RENDER
// Rendering array.
// TODO: Fix variable names.
__device__ int *device_render_cells;
int *host_render_cells;
int *d_device_render_cells;
#endif  // OPTION_RENDER

// Dataset.
__device__ int SIZE_X;
__device__ int SIZE_Y;
__managed__ CellV **cells;
__managed__ Cell *cells2;
dataset_t dataset;

// Only count alive agents in state 0.
__device__ int num_alive_neighbors(AgentV *ptr) {
    int cell_x = ptr->cell_id_ % SIZE_X;
    int cell_y = ptr->cell_id_ / SIZE_X;
    int result = 0;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                // CONCORD
                AgentV *ptr;
                CONCORD(ptr, cells[ny * SIZE_X + nx], agent());

                AgentV *alive = nullptr;
                if (ptr) {
                    // CONCORD
                    bool cond;
                    CONCORD(cond, ptr, isAlive());
                    if (cond)

                        // CONCORD
                        CONCORD(alive, cells[ny * SIZE_X + nx], agent());
                }

                // CONCORD
                bool cond2 =false;
                if (alive != nullptr) {
                    CONCORD(cond2, alive, is_state_equal(0));
                }
                if (alive != nullptr && cond2) {
                    result++;
                }
            }
        }
    }

    return result;
}

__device__ void create_candidates(AgentV *ptr) {
    // CONCORD
    bool cond;
    CONCORD(cond, ptr, is_new());
    assert(cond);

    // TODO: Consolidate with Agent::num_alive_neighbors().
    int cell_x = ptr->cell_id_ % SIZE_X;
    int cell_y = ptr->cell_id_ / SIZE_X;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                auto cid = ny * SIZE_X + nx;
                // CONCORD
                bool cond2;
                CONCORD(cond2, cells[cid], is_empty())
                if (cond2) {
                    if (atomicCAS(&cells[cid]->reserved, 0, 1) == 0) {
                        // CONCORD
                        CONCORD(cells[cid],
                                set_agent(cid, AgentType::isCandidate));
                        ;
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
        // CONCORD
        CONCORD(cells[cell_ids[i]], set_agent(cell_ids[i], AgentType::isAlive));
        ;
        // CONCORD
        AgentV *ptr;
        CONCORD(ptr, cells[cell_ids[i]], agent());
        assert(ptr != nullptr);
        // CONCORD
        CONCORD(ptr, cells[cell_ids[i]], agent());
        int id;
        CONCORD(id, ptr, cell_id());
        assert(id == cell_ids[i]);
    }
}

__global__ void update() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        // CONCORD
        AgentV *ptr;
        CONCORD(ptr, cells[i], agent());
        ;
        if (ptr) {
            // CONCORD
            bool cond;
            CONCORD(cond, ptr, isCandidate())
            if (cond) {
                int cid = ptr->cell_id_;

                // CONCORD
                int act;
                CONCORD(act, ptr, get_action());
                int act2;
                CONCORD(act2, ptr, get_action());
                if (act == kActionSpawnAlive) {
                    // CONCORD
                    CONCORD(cells[cid], set_agent(cid, AgentType::isAlive));

                    // CONCORD
                } else if (act2 == kActionDie) {
                    // CONCORD
                    CONCORD(cells[cid], delete_agent());

                    cells[cid]->reserved = 0;
                }
            }
        }
    }
}

__global__ void prepare() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        // CONCORD
        AgentV *ptr;
        CONCORD(ptr, cells[i], agent());
        ;
        if (ptr) {
            // CONCORD
            bool cond;
            CONCORD(cond, ptr, isAlive());
            if (cond) {
                // CONCORD
                bool cond2;
                CONCORD(cond2, ptr, is_state_equal(0));
                if (cond2) {
                    // CONCORD
                    CONCORD(ptr, set_is_new(false));

                    // Also counts this object itself.
                    int alive_neighbors = num_alive_neighbors(ptr) - 1;

                    const bool stay_alive_param[9] = kStayAlive;
                    if (!stay_alive_param[alive_neighbors]) {
                        // CONCORD
                        CONCORD(ptr, set_action(kActionDie));
                    }
                }
            }
        }
        // CONCORD
        CONCORD(ptr, cells[i], agent());

        if (ptr) {
            // CONCORD
            bool cond;
            CONCORD(cond, ptr, isCandidate());
            if (cond) {
                int alive_neighbors = num_alive_neighbors(ptr);
                const bool spawn_param[9] = kSpawnNew;

                if (spawn_param[alive_neighbors]) {
                    // CONCORD
                    CONCORD(ptr, set_action(kActionSpawnAlive));
                    ;
                } else if (alive_neighbors == 0) {
                    // CONCORD
                    CONCORD(ptr, set_action(kActionDie));
                    ;
                }
            }
        }
    }
}

__global__ void update_checksum() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        // CONCORD
        AgentV *ptr;
        CONCORD(ptr, cells[i], agent());
        ;
        // CONCORD
        if (ptr) ptr->update_checksum();
    }
}

__global__ void alive_update() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        // CONCORD
        AgentV *ptr;
        CONCORD(ptr, cells[i], agent());
        ;
        if (ptr) {
            // CONCORD
            bool cond;
            CONCORD(cond, ptr, isAlive());
            if (cond) {
                int cid = ptr->cell_id_;

                // TODO: Consider splitting in two classes for less divergence.
                // CONCORD
                bool cond2;
                CONCORD(cond2, ptr, is_new());
                if (cond2) {
                    // Create candidates in neighborhood.
                    create_candidates(ptr);
                } else {
                    // CONCORD
                    int act;
                    bool flag1;
                    CONCORD(act, ptr, get_action());
                    flag1 = act == kActionDie;

                    // CONCORD
                    bool cond;
                    CONCORD(cond, ptr, is_state_equal(0));
                    flag1 = flag1 && cond;

                    // CONCORD
                    bool flag2;
                    CONCORD(flag2, ptr, is_state_in_range(0, kNumStates));

                    // CONCORD
                    bool flag3;
                    CONCORD(flag3, ptr, is_state_equal(kNumStates));
                    ;
                    if (flag1) {
                        // Increment state. If reached max. state, replace with
                        // Candidate.

                        // CONCORD
                        CONCORD(ptr, inc_state());

                        // CONCORD
                        CONCORD(ptr, set_action(kActionNone));

                    } else if (flag2) {
                        // CONCORD
                        CONCORD(ptr, inc_state());

                    } else if (flag3) {
                        // Replace with Candidate.

                        // CONCORD
                        CONCORD(cells[cid],
                                set_agent(cid, AgentType::isCandidate));
                        ;
                        // delete this;
                    }
                }
            }
        }
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

__device__ __noinline__ void AgentV::update_checksum() {
    // CONCORD
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

int main(int /*argc*/, char ** /*argv*/) {
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

    cudaMalloc(&cells, sizeof(Cell *) * dataset.x * dataset.y);
    cudaMalloc(&cells2, sizeof(Cell) * dataset.x * dataset.y);

    // Initialize cells.
    create_cells<<<1024, 1024>>>();
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
