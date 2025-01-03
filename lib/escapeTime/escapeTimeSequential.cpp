#include "escapeTimeSequential.h"
#include "escapeTime/coloring.h"

int pointEscapeTime(double cReal, double cImaginary, int maxIters) {
	double zReal = 0;
	double zImaginary = 0;

	double zRealSqr = 0;
	double zImaginarySqr = 0;

	//if the absolute value of z ever exceeds 2, it is guaranteed to escape the mandelbrot set. Therefore, this loop can be terminated
	int i;
	for (i = 0; i < maxIters && zRealSqr + zImaginarySqr <= 4; i++) {
		//(ai + b) ^2 = -a^2 + 2abi + b^2
		zImaginary = 2 * zReal * zImaginary + cImaginary;
		zReal = zRealSqr - zImaginarySqr + cReal;

		zRealSqr = zReal * zReal;
		zImaginarySqr = zImaginary * zImaginary;
	}
	
	return i;
}

void escapeTimeSequential(std::vector<RGBColor>& escapeTimes, uint maxIters, uint sizeX, uint sizeY, double scale, double panX, double panY, ColoringFunctionType func) {
	for (int x = 0; x < sizeX; x++) {
		for (int y = 0; y < sizeY; y++) {
			double real = ((double)x) * scale + panX;
			double imaginary = ((double)y) * scale + panY;
			auto escapeIterations = pointEscapeTime(real, imaginary, maxIters);
			escapeTimes[x * sizeY + y] = colorFunction(escapeIterations, maxIters, func);
		}
	}
}
