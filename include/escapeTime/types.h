#ifndef MANDELBROT_ALGORITHM_TYPES
#define MANDELBROT_ALGORITHM_TYPES

//The entry point function for any version of the escape time algorithm should be the same type of function so that they are interoperable.
typedef void escapeTimeAlgorithm(int* escapeTimes, int max_iters, int sizeX, int sizeY, double scale, double panX, double panY);

#endif