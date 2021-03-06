#ifndef EXAMPLE_GENERATION_CONFIGURATION_H
#define EXAMPLE_GENERATION_CONFIGURATION_H

#include "dataset_loader.h"

extern dataset_t dataset;

// TODO: Should be constants instead of macros.
//#define kNumStates 50
//#define kStayAlive { false, false, true, false, false, false, false, false }
//#define kSpawnNew  { false, true, false, true, false, false, false, false }

#define kNumStates 255
#define kStayAlive { true, false, true, true, false, true, true, true, true }
#define kSpawnNew  { false, false, false, true, true, false, true, false, true }


//#define PARAM_NUM_ITER 1
#ifdef PARAM_NUM_ITER
static const int kNumIterations = PARAM_NUM_ITER;
#else
static const int kNumIterations = 1000;  // 1000
#endif  // PARAM_NUM_ITER

#ifndef PARAM_SIZE
static const int kSize = 1024;  // 18000
#else
static const int kSize = PARAM_SIZE;
#endif  // PARAM_SIZE

#ifndef PARAM_MAX_OBJ
static const int kNumObjects = 2147483647/100;//;48*64*64*64*64;
#else
static const int kNumObjects = PARAM_MAX_OBJ;
#endif  // PARAM_MAX_OBJ

#endif  // EXAMPLE_GENERATION_CONFIGURATION_H

