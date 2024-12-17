#ifndef ESCAPE_TIME_SEQUENTIAL
#define ESCAPE_TIME_SEQUENTIAL
#include <vector>
#include "escapeTime/coloring.h"
#include "coloringFunctionsHost.h"

void escapeTimeSequential(std::vector<RGBColor>& escapeTimes, int maxIters, int sizeX, int sizeY, double scale, double panX, double panY, ColoringFunction::Function func);

int pointEscapeTime(double cReal, double cImaginary, int maxIters);

#endif 