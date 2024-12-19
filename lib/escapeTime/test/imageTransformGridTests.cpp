#include <gtest/gtest.h>

#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeSequential.h"

namespace cudaGrid {
	TEST(testCudaGrid, InitializeGrid) {
		auto dim = 5;
		MandelbrotImageTransformGrid grid(dim, dim, 100, 1.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dim * dim);
		escapeTimeSequential(rawGrid, 100, dim, dim, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dim; i++) {
			for (int j = 0 ; j < dim; j++) {
				EXPECT_EQ(rawGrid[i * dim + j], grid.get(i, j));
			}
		}

	}

	TEST(testCudaGrid, InitializeLargeGrid) {
		auto dim = 1000;
		MandelbrotImageTransformGrid grid(dim, dim, 100, 1.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dim * dim);
		escapeTimeSequential(rawGrid, 100, dim, dim, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				EXPECT_EQ(rawGrid[i * dim + j], grid.get(i, j));
			}
		}

	}

	TEST(testCudaGrid, InitializeRectangleGrid) {
		auto dimx = 1000;
		auto dimy = 500;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}

	}

}