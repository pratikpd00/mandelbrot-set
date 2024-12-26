#include <gtest/gtest.h>

#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeSequential.h"
#include "escapeTimeCudaWrapper.h"

namespace ImageTransformGridTests {
	TEST(testCudaGrid, InitializeGrid) {
		auto dim = 5;
		MandelbrotImageTransformGrid grid(dim, dim, 100, 0.1, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dim * dim);
		escapeTimeSequential(rawGrid, 100, dim, dim, 0.1, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dim; i++) {
			for (int j = 0 ; j < dim; j++) {
				EXPECT_EQ(rawGrid[i * dim + j], grid.get(i, j));
			}
		}

	}

	TEST(testCudaGrid, InitializeLargeGrid) {
		auto dim = 1000;
		MandelbrotImageTransformGrid grid(dim, dim, 100, 0.1, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dim * dim);
		escapeTimeSequential(rawGrid, 100, dim, dim, 0.1, 0.0, 0.0, ColoringFunctionType::DEFAULT);

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

	TEST(testCudaGrid, InitializeWithParameters) {
		auto dimx = 1000;
		auto dimy = 500;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 4.0, 0.5, -0.5);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 4.0, 0.5, -0.5, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}

	}

	//write a test for the zoom method
	TEST(testCudaGrid, Zoom) {
		auto dimx = 1000;
		auto dimy = 500;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 2.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCudaGrid, MultiZoom) {
		auto dimx = 1000;
		auto dimy = 500;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.0, 0.0, 0.0);
		grid.zoom(3.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 3.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCudaGrid, OffsetZoom) {
		auto dimx = 20;
		auto dimy = 10;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.5, 20, 10);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeCUDA(rawGrid.data(), 100, dimx, dimy, 2.5, -30.0, -15.0);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}


	TEST(testCudaGrid, Pan) {
		auto dimx = 20;
		auto dimy = 10;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 0.1, 0.0, 0.0);
		grid.translate(-1, -2);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeCUDA(rawGrid.data(), 100, dimx, dimy, 0.1, 0.1, 0.2);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}


	TEST(testCudaGrid, MultiPan) {
		auto dimx = 20;
		auto dimy = 10;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 0.1, 0.0, 0.0);
		grid.translate(-1, -2);
		grid.translate(-1, 4);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeCUDA(rawGrid.data(), 100, dimx, dimy, 0.1, 0.2, -0.2);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCudaGrid, Resize) {
		auto dimx = 1000;
		auto dimy = 500;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.resizeGrid(100, 100);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, 100, 100, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < 100; i++) {
			for (int j = 0; j < 100; j++) {
				EXPECT_EQ(rawGrid[i * 100 + j], grid.get(i, j));
			}
		}
	}

	TEST(testCudaGrid, ResizeLarger) {
		auto dimx = 1000;
		auto dimy = 500;
		MandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.resizeGrid(600, 1100);
		std::vector<RGBColor> rawGrid(600 * 1100);
		escapeTimeSequential(rawGrid, 100, 600, 1100, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < 600; i++) {
			for (int j = 0; j < 1100; j++) {
				EXPECT_EQ(rawGrid[i * 1100 + j], grid.get(i, j));
			}
		}
	}


}