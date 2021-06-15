#include <chrono>

#include "../configuration.h"
#include "../dataset_loader.h"

#include "gol.h"

#ifdef OPTION_RENDER
// Rendering array.
// TODO: Fix variable names.
__device__ char *device_render_cells;
char *host_render_cells;
char *d_device_render_cells;
#endif  // OPTION_RENDER

// Dataset.
__device__ int SIZE_X;
__device__ int SIZE_Y;
__managed__ CellV **cells;
dataset_t dataset;

__device__ int num_alive_neighbors(AgentV *ptr) {
    void **vtable;

    COAL_AgentV_cell_id(ptr);
    int cell_x = CLEANPTR( ptr , AgentV*)->cell_id() % SIZE_X;
    COAL_AgentV_cell_id(ptr);
    int cell_y = CLEANPTR( ptr , AgentV*)->cell_id() / SIZE_X;
    int result = 0;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                COAL_CellV_agent(cells[ny * SIZE_X + nx]);
                AgentV *tmp = CLEANPTR( cells[ny * SIZE_X + nx] , CellV*)->agent();

                if (CLEANPTR( tmp , AgentV*)) {
                    COAL_AgentV_isAlive(tmp);
                    if (CLEANPTR( tmp , AgentV*)->isAlive()) {
                        result++;
                    }
                }
            }
        }
    }

    return result;
}

__device__ void maybe_create_candidate(AgentV *ptr, int x, int y) {
    void **vtable;
    // Check neighborhood of cell to determine who should create Candidate.
    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                COAL_CellV_agent(cells[ny * SIZE_X + nx]);
                AgentV *alive = CLEANPTR( cells[ny * SIZE_X + nx] , CellV*)->agent();
                if (alive != nullptr) {
                    COAL_AgentV_is_new(alive);
                    if (CLEANPTR( alive , AgentV*)->is_new()) {
                        if (alive == ptr) {
                            // Create candidate now.
                            COAL_CellV_set_agent(cells[y * SIZE_X + x]);
                            CLEANPTR( cells[y * SIZE_X + x] , CellV*)->set_agent(
                                (y * SIZE_X + x), AgentType::isCandidate);

                        }  // else: Created by other thread.

                        return;
                    }
                }
            }
        }
    }

    assert(false);
}
__device__ void create_candidates(AgentV *ptr) {
    void **vtable;
    COAL_AgentV_is_new(ptr);
    assert(CLEANPTR( ptr , AgentV*)->is_new());
    COAL_AgentV_isAlive(ptr);
     assert(CLEANPTR( ptr , AgentV*)->isAlive());
    // TODO: Consolidate with Agent::num_alive_neighbors().
    COAL_AgentV_cell_id(ptr);
    int cell_x = CLEANPTR( ptr , AgentV*)->cell_id() % SIZE_X;
    COAL_AgentV_cell_id(ptr);
    int cell_y = CLEANPTR( ptr , AgentV*)->cell_id() / SIZE_X;

    for (int dx = -1; dx < 2; ++dx) {
        for (int dy = -1; dy < 2; ++dy) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
                COAL_CellV_is_empty(cells[ny * SIZE_X + nx]);
                if (CLEANPTR( cells[ny * SIZE_X + nx] , CellV*)->is_empty()) {
                    // Candidate should be created here.
                    maybe_create_candidate(ptr, nx, ny);
                }
            }
        }
    }
}

__device__ void Alive_prepare(AgentV *ptr) {
    void **vtable;

    if (CLEANPTR( ptr , AgentV*)) {
        COAL_AgentV_isAlive(ptr);
        if (CLEANPTR( ptr , AgentV*)->isAlive()) {
            COAL_AgentV_set_is_new(ptr);
            CLEANPTR( ptr , AgentV*)->set_is_new(false);

            // Also counts this object itself.
            int alive_neighbors = num_alive_neighbors(ptr) - 1;

            if (alive_neighbors < 2 || alive_neighbors > 3) {
                COAL_AgentV_set_action(ptr);
                CLEANPTR( ptr , AgentV*)->set_action(kActionDie);
            }
        }
    }
}

__device__ void Alive_update(AgentV *ptr) {
    void **vtable;
    if (CLEANPTR( ptr , AgentV*)) {
        COAL_AgentV_isAlive(ptr);

        if (CLEANPTR( ptr , AgentV*)->isAlive()) {
            COAL_AgentV_cell_id(ptr);

            int cid = CLEANPTR( ptr , AgentV*)->cell_id();

            // TODO: Consider splitting in two classes for less divergence.
            COAL_AgentV_is_new(ptr);
            if (CLEANPTR( ptr , AgentV*)->is_new()) {
                // Create candidates in neighborhood.
                create_candidates(ptr);
            } else {
                COAL_AgentV_get_action(ptr);
                if (CLEANPTR( ptr , AgentV*)->get_action() == kActionDie) {
                    COAL_CellV_set_agent(cells[cid]);
                    CLEANPTR( cells[cid] , CellV*)->set_agent(cid, AgentType::isCandidate);
                }
            }
        }
    }
}

__global__ void alive_prepare() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        AgentV *ptr = CLEANPTR( cells[i] , CellV*)->agent();
        Alive_prepare(ptr);
    }
}

__global__ void alive_update() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        AgentV *ptr = CLEANPTR( cells[i] , CellV*)->agent();
        Alive_update(ptr);
    }
}
__device__ void Candidate_prepare(AgentV *ptr) {
    void **vtable;
    if (CLEANPTR( ptr , AgentV*)) {
        COAL_AgentV_isCandidate(ptr);
        if (CLEANPTR( ptr , AgentV*)->isCandidate()) {
            int alive_neighbors = num_alive_neighbors(ptr);

            if (alive_neighbors == 3) {
                COAL_AgentV_set_action(ptr);
                CLEANPTR( ptr , AgentV*)->set_action(kActionSpawnAlive);

            } else if (alive_neighbors == 0) {
                COAL_AgentV_set_action(ptr);
                CLEANPTR( ptr , AgentV*)->set_action(kActionDie);
            }
        }
    }
}

__device__ void Candidate_update(AgentV *ptr) {
    // TODO: Why is this necessary?
    void **vtable;

    if (CLEANPTR( ptr , AgentV*)) {
        COAL_AgentV_isCandidate(ptr);

        if (CLEANPTR( ptr , AgentV*)->isCandidate()) {
            COAL_AgentV_cell_id(ptr);

            int cid = CLEANPTR( ptr , AgentV*)->cell_id();
            COAL_AgentV_get_action(ptr);
            if (CLEANPTR( ptr , AgentV*)->get_action() == kActionSpawnAlive) {
              COAL_CellV_set_agent(cells[cid]);
                CLEANPTR( cells[cid] , CellV*)->set_agent(cid, AgentType::isAlive);
            } else {
              COAL_AgentV_get_action(ptr);
                if (CLEANPTR( ptr , AgentV*)->get_action() == kActionDie) {
                  COAL_CellV_delete_agent(cells[cid]);
                    CLEANPTR( cells[cid] , CellV*)->delete_agent();
                }
            }
        }
    }
}

__global__ void candidate_prepare() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        AgentV *ptr = CLEANPTR( cells[i] , CellV*)->agent();
        Candidate_prepare(ptr);
    }
}

__global__ void candidate_update() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        AgentV *ptr = CLEANPTR( cells[i] , CellV*)->agent();
        Candidate_update(ptr);
    }
}


// Must be followed by Alive::update().
__global__ void load_game(int *cell_ids, int num_cells) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_cells;
         i += blockDim.x * gridDim.x) {
        CLEANPTR( cells[cell_ids[i]] , CellV*)->set_agent(cell_ids[i], AgentType::isAlive);
        assert(CLEANPTR(CLEANPTR( cells[cell_ids[i]] , CellV*)->agent(),AgentV *)->cell_id() == cell_ids[i]);
    }
}
int checksum();
void transfer_dataset() {
    int *dev_cell_ids;
    cudaMalloc(&dev_cell_ids, sizeof(int) * dataset.num_alive);
    cudaMemcpy(dev_cell_ids, dataset.alive_cells,
               sizeof(int) * dataset.num_alive, cudaMemcpyHostToDevice);

#ifndef NDEBUG
    printf("Loading on GPU: %i alive cells.\n", dataset.num_alive);
#endif  // NDEBUG

    load_game<<<128, 128>>>(dev_cell_ids, dataset.num_alive);
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(dev_cell_ids);

    alive_update<<<1024, 1024>>>();
    gpuErrchk(cudaDeviceSynchronize());
}

__device__ int device_checksum;
__device__ int device_num_candidates;

__device__ void Agent::update_checksum() {
    if (this->isAlive())
        atomicAdd(&device_checksum, 1);
    else if (this->isCandidate())
        atomicAdd(&device_num_candidates, 1);
}
__global__ void update_checksum() {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
         i += blockDim.x * gridDim.x) {
        AgentV *ptr = CLEANPTR( cells[i] , CellV*)->agent();
        if (CLEANPTR( ptr , AgentV*)) CLEANPTR( ptr , AgentV*)->update_checksum();
    }
}
int checksum() {
    int host_checksum = 0;
    int host_num_candidates = 0;
    cudaMemcpyToSymbol(device_checksum, &host_checksum, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device_num_candidates, &host_num_candidates, sizeof(int),
                       0, cudaMemcpyHostToDevice);

    // allocator_handle->parallel_do<Alive, &Alive::update_checksum>();
    // allocator_handle->parallel_do<Candidate, &Candidate::update_counter>();
    update_checksum<<<1024, 1024>>>();
    cudaMemcpyFromSymbol(&host_checksum, device_checksum, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_num_candidates, device_num_candidates,
                         sizeof(int), 0, cudaMemcpyDeviceToHost);

    return host_checksum;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s filename.pgm\n", argv[0]);
        exit(1);
    } else {
        // Load data set.
        dataset = load_from_file(argv[1]);
    }
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);
    mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
    obj_alloc my_obj_alloc(&shared_mem, atoll(argv[2]));
    cudaMemcpyToSymbol(SIZE_X, &dataset.x, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(SIZE_Y, &dataset.y, sizeof(int), 0,
                       cudaMemcpyHostToDevice);

    // Allocate memory.

    // cudaMalloc(&host_cells, sizeof(Cell *) * dataset.x * dataset.y);
    // cudaMemcpyToSymbol(cells, &host_cells, sizeof(Cell **), 0,
    //                    cudaMemcpyHostToDevice);

    cells = (CellV **)my_obj_alloc.calloc<CellV *>(dataset.x * dataset.y);
    for (int i = 0; i < dataset.x * dataset.y; i++) {
        cells[i] = (Cell *)my_obj_alloc.my_new<Cell>();
        CLEANPTR(cells[i],CellV*)->inst_cell(&my_obj_alloc);
        // assert(cells[i] != nullptr);
    }
    printf("DONE\n");
    my_obj_alloc.toDevice();
    my_obj_alloc.create_table();
    vfun_table = my_obj_alloc.get_vfun_table();
    // Initialize cells.
    // create_cells<<<128, 128>>>();
    gpuErrchk(cudaDeviceSynchronize());
 
    transfer_dataset();

    auto time_start = std::chrono::system_clock::now();
    printf("Checksum: %i\n", checksum());
    // Run simulation.
    for (int i = 0; i < kNumIterations; ++i) {
        candidate_prepare<<<1024, 1024>>>();
        gpuErrchk(cudaDeviceSynchronize());

        alive_prepare<<<1024, 1024>>>();
        gpuErrchk(cudaDeviceSynchronize());

        candidate_update<<<1024, 1024>>>();
        gpuErrchk(cudaDeviceSynchronize());

        alive_update<<<1024, 1024>>>();
        gpuErrchk(cudaDeviceSynchronize());

        // printf("Checksum: %i\n", checksum());
    }

    auto time_end = std::chrono::system_clock::now();
    auto elapsed = time_end - time_start;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    printf("Checksum: %i\n", checksum());

    printf("%lu, \n", micros);

    return 0;
}
