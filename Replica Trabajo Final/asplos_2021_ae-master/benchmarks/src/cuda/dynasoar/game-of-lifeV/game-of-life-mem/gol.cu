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
#endif // OPTION_RENDER

// Dataset.
__device__ int SIZE_X;
__device__ int SIZE_Y;
__managed__ CellV **cells;
dataset_t dataset;

__device__ int num_alive_neighbors(AgentV *ptr) {
  int cell_x = ptr->cell_id() % SIZE_X;
  int cell_y = ptr->cell_id() / SIZE_X;
  int result = 0;

  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        AgentV *tmp = cells[ny * SIZE_X + nx]->agent();
       
        if (tmp) {
          if (tmp->isAlive()) {

            result++;
          }
        }
      }
    }
  }

  return result;
}

__device__ void maybe_create_candidate(AgentV *ptr, int x, int y) {
  // Check neighborhood of cell to determine who should create Candidate.
  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = x + dx;
      int ny = y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        AgentV *alive = cells[ny * SIZE_X + nx]->agent();
        if (alive != nullptr) {
          if (alive->is_new()) {
            if (alive == ptr) {
              // Create candidate now.
              cells[y * SIZE_X + x]->set_agent((y * SIZE_X + x),
                                               AgentType::isCandidate);

            } // else: Created by other thread.

            return;
          }
        }
      }
    }
  }

  assert(false);
}
__device__ void create_candidates(AgentV *ptr) {
  assert(ptr->is_new());
  assert(ptr->isAlive());
  // TODO: Consolidate with Agent::num_alive_neighbors().
  int cell_x = ptr->cell_id() % SIZE_X;
  int cell_y = ptr->cell_id() / SIZE_X;

  for (int dx = -1; dx < 2; ++dx) {
    for (int dy = -1; dy < 2; ++dy) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;

      if (nx > -1 && nx < SIZE_X && ny > -1 && ny < SIZE_Y) {
        if (cells[ny * SIZE_X + nx]->is_empty()) {
          // Candidate should be created here.
          maybe_create_candidate(ptr, nx, ny);
        }
      }
    }
  }
}

__device__ void Alive_prepare(AgentV *ptr) {
  if (ptr) {
    if (ptr->isAlive()) {
      ptr->set_is_new(false);

      // Also counts this object itself.
      int alive_neighbors = num_alive_neighbors(ptr) - 1;

      if (alive_neighbors < 2 || alive_neighbors > 3) {
        ptr->set_action(kActionDie);
      }
    }
  }
}

__device__ void Alive_update(AgentV *ptr) {
  if (ptr) {
    if (ptr->isAlive()) {
      int cid = ptr->cell_id();

      // TODO: Consider splitting in two classes for less divergence.
      if (ptr->is_new()) {
        // Create candidates in neighborhood.
        create_candidates(ptr);
      } else {
        if (ptr->get_action() == kActionDie) {
          cells[cid]->set_agent(cid, AgentType::isCandidate);
        }
      }
    }
  }
}

__global__ void alive_prepare() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
       i += blockDim.x * gridDim.x) {
    AgentV *ptr = cells[i]->agent();
    Alive_prepare(ptr);
  }
}

__global__ void alive_update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
       i += blockDim.x * gridDim.x) {
    AgentV *ptr = cells[i]->agent();
    Alive_update(ptr);
  }
}
__device__ void Candidate_prepare(AgentV *ptr) {
  if (ptr) {
    if (ptr->isCandidate()) {
      int alive_neighbors = num_alive_neighbors(ptr);

      if (alive_neighbors == 3) {
        ptr->set_action(kActionSpawnAlive);

      } else if (alive_neighbors == 0) {
        ptr->set_action(kActionDie);
      }
    }
  }
}

__device__ void Candidate_update(AgentV *ptr) {
  // TODO: Why is this necessary?
  if (ptr) {
    if (ptr->isCandidate()) {
      int cid = ptr->cell_id();

      if (ptr->get_action() == kActionSpawnAlive) {
        cells[cid]->set_agent(cid, AgentType::isAlive);
      } else if (ptr->get_action() == kActionDie) {
        cells[cid]->delete_agent();
      }
    }
  }
}

__global__ void candidate_prepare() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
       i += blockDim.x * gridDim.x) {
    AgentV *ptr = cells[i]->agent();
    Candidate_prepare(ptr);
  }
}

__global__ void candidate_update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
       i += blockDim.x * gridDim.x) {
    AgentV *ptr = cells[i]->agent();
    Candidate_update(ptr);
  }
}
__global__ void create_cells() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
       i += blockDim.x * gridDim.x) {
    cells[i] = new Cell();
    assert(cells[i] != nullptr);
  }
}

// Must be followed by Alive::update().
__global__ void load_game(int *cell_ids, int num_cells) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_cells;
       i += blockDim.x * gridDim.x) {
    cells[cell_ids[i]]->set_agent(cell_ids[i], AgentType::isAlive);
    assert(cells[cell_ids[i]]->agent()->cell_id() == cell_ids[i]);
  }
}
int checksum();
void transfer_dataset() {
  int *dev_cell_ids;
  cudaMalloc(&dev_cell_ids, sizeof(int) * dataset.num_alive);
  cudaMemcpy(dev_cell_ids, dataset.alive_cells, sizeof(int) * dataset.num_alive,
             cudaMemcpyHostToDevice);

#ifndef NDEBUG
  printf("Loading on GPU: %i alive cells.\n", dataset.num_alive);
#endif // NDEBUG

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
  else if(this-> isCandidate())
    atomicAdd(&device_num_candidates, 1);
}
__global__ void update_checksum() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE_X * SIZE_Y;
       i += blockDim.x * gridDim.x) {
    AgentV *ptr = cells[i]->agent();
    if (ptr)
      ptr->update_checksum();
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
  cudaMemcpyFromSymbol(&host_num_candidates, device_num_candidates, sizeof(int),
                       0, cudaMemcpyDeviceToHost);

  return host_checksum  ;
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
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  cells = (CellV **)my_obj_alloc.calloc<CellV *>(dataset.x * dataset.y);
  for (int i = 0; i < dataset.x * dataset.y; i++) {
      cells[i] = (Cell *)my_obj_alloc.my_new<Cell>();
      cells[i]->inst_cell(&my_obj_alloc);
      // assert(cells[i] != nullptr);
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  my_obj_alloc.toDevice();
  high_resolution_clock::time_point t3 = high_resolution_clock::now();
  duration<double> alloc_time = duration_cast<duration<double>>(t2 - t1);
  duration<double> vptr_time = duration_cast<duration<double>>(t3 - t2);

  printf("alloc_time : %f \nvptr patching : %f \n",alloc_time.count(),vptr_time.count() );
  printf("number of objs:%d\n", dataset.x * dataset.y);
  // Initialize cells.
  //create_cells<<<128, 128>>>();
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
