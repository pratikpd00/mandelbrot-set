#include "escapeTime/coloring.h"
#include "coloringFunctions.cuh"

namespace ColoringFunction {
    
    __device__ __host__ RGBColor Default(int iterations, int maxIterations) {
        return iterations;
    }


    __device__ __host__ RGBColor color(int iterations, int maxIterations, ColoringFunction::Function function) {
        //Write a switch case for variable function that returns the result of the corresponding coloring function
        switch (function)
        {
        case ColoringFunction::Function::DEFAULT:
            return Default(iterations, maxIterations);
            break;
        
        default:
            return Default(iterations, maxIterations);
            break;
        }

    }
}
