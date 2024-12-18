//Same header guard as the host header because only one of the two should be included
#ifndef COLORING_FUNCTIONS_H
#define COLORING_FUNCTIONS_H

#include "escapeTime/coloring.h"



__device__ __host__ RGBColor colorFunction(int iterations, int maxIterations, ColoringFunctionType function);

#endif