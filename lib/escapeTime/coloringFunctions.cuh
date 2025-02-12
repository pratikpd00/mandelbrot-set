//Same header guard as the host header because only one of the two should be included
#ifndef COLORING_FUNCTIONS_H
#define COLORING_FUNCTIONS_H

#include "escapeTime/coloring.h"



__device__ __host__ RGBColor colorFunction(int iterations, int maxIterations, ColoringFunctionType function);

__device__ __host__ inline RGBColor deviceColor(uint8_t r, uint8_t g, uint8_t b) {
    return ALPHA_OPAQUE | (r << 16) | (g << 8) | b;
}

#endif