#ifndef ESCAPE_TIME_CUDA_WRAPPER_H
#define ESCAPE_TIME_CUDA_WRAPPER_H

#include <escapeTime/coloring.h>

void escapeTimeCUDA(RGBColor* escapeTimes, int maxIters, int sizeX, int sizeY, double scale, double panX, double panY);

#endif