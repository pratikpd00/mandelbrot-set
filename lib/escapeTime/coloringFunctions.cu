#include "escapeTime/coloring.h"
#include "coloringFunctions.cuh"

namespace coloringFunction {
    __device__ __host__ RGBColor Color(int iterations, int maxIterations, ColoringFunction::Function function) {
        //Write a switch case for variable function that returns the result of the corresponding coloring function
        switch (function)
        {
        case ColoringFunction::DEFAULT:
            return Default(iterations, maxIterations);
            break;
        
        default:
            return Default(iterations, maxIterations);
            break;
        }

    }

    __device__ __host__ RGBColor Default(int iterations, int maxIterations) {
        return iterations;
    }
}