#include <gtest/gtest.h>

#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeSequential.h"

namespace cudaGrid {
	TEST(testCudaGrid, InitializeGrid) {
		MandelbrotImageTransformGrid grid(1000, 1000, 100, 1.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(1000);
		escapeTimeSequential(rawGrid, 100, 1000*1000, 1.0, 0.0, 0.0, 1.0, 1.0, ColoringFunctionType::DEFAULT);

	}

}