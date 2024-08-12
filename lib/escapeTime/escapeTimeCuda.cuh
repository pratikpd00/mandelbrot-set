#ifndef ESCAPE_TIME_CUDA
#define ESCAPE_TIME_CUDA

#include "escapeTime/escapeTimeCuda.h"
__global__ void escapeTime(int** escapeTimes, int max_iters, double scale, double panX, double panY);

#endif 
