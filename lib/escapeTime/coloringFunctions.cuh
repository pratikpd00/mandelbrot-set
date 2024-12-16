#include "escapeTime/coloring.h"


namespace ColoringFunction {
    __device__ __host__ RGBColor Color(int iterations, int maxIterations, ColoringFunction::Function function);

    __device__ __host__ RGBColor Default(int iterations, int maxIterations);
}