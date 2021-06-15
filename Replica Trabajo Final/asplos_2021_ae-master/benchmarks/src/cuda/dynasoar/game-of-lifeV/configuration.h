#ifndef EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
#define EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H

#include "dataset_loader.h"
#include <ctime>
using CellT = int;

extern dataset_t dataset;

//#define PARAM_NUM_ITER 1
#ifdef PARAM_NUM_ITER
static const int kNumIterations = PARAM_NUM_ITER;
#else
static const int kNumIterations = 3000;  // 10000
#endif  // PARAM_NUM_ITER

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_GAME_OF_LIFE_SOA_CONFIGURATION_H
