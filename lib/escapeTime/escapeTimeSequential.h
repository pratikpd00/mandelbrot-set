#ifndef ESCAPE_TIME_SEQUENTIAL
#define ESCAPE_TIME_SEQUENTIAL
#include <vector>
#include "escapeTime/coloring.h"

void escapeTimeSequential(std::vector<RGBColor>& escapeTimes, int maxIters, int sizeX, int sizeY, double scale, double panX, double panY);

int pointEscapeTime(double cReal, double cImaginary, int maxIters);

#endif 