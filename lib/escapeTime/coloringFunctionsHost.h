//Same header guard as the device header because only one of the two should be included
#ifndef COLORING_FUNCTIONS_H
#define COLORING_FUNCTIONS_H

#include "escapeTime/coloring.h"

RGBColor colorFunction(int iterations, int maxIterations, ColoringFunctionType function);

#endif
