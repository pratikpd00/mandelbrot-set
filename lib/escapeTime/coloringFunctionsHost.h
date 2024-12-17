//Same header guard as the device header because only one of the two should be included
#ifndef COLORING_FUNCTIONS_H
#define COLORING_FUNCTIONS_H

#include "escapeTime/coloring.h"

namespace ColoringFunction {
    RGBColor color(int iterations, int maxIterations, ColoringFunction::Function function);
}

#endif
