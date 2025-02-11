#include "escapeTime/mandelbrotImageTransformGrid.h"

int main() {
	ImageTransformGrid* grid = new CudaMandelbrotImageTransformGrid(2000, 1000, 200, 0.005, -2, -1.5);
	grid->zoom(0.5, 128, 96);
	delete grid;
	return 0;
}