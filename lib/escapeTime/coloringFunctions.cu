#include "escapeTime/coloring.h"
#include "coloringFunctions.cuh"


__device__ __host__ RGBColor Default(int iterations, int maxIterations) {
    return iterations;
}

__device__ __host__ RGBColor Blue(int iterations, int maxIterations) {
    int weight = iterations * 255 / maxIterations;
    return deviceColor(0, 0, weight);

}

__device__ __host__ RGBColor colorFunction(int iterations, int maxIterations, ColoringFunctionType function) {
    //Write a switch case for variable function that returns the result of the corresponding coloring function
    switch (function)
    {
    case ColoringFunctionType::DEFAULT:
        return Default(iterations, maxIterations);
        break;
    case ColoringFunctionType::BLUE:
		return Blue(iterations, maxIterations);
		break;
    default:
        return Default(iterations, maxIterations);
        break;
    }

}
