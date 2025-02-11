#include <gtest/gtest.h>

#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeSequential.h"
#include "escapeTimeCudaWrapper.h"

namespace ImageTransformGridTests {
	TEST(testCudaGrid, InitializeGrid) {
		auto dim = 5;
		CudaMandelbrotImageTransformGrid grid(dim, dim, 100, 0.1, 0.0, 0.0);
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
		CudaMandelbrotImageTransformGrid grid(dim, dim, 100, 0.1, 0.0, 0.0);
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
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
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
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 4.0, 0.5, -0.5);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 4.0, 0.5, -0.5, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}

	}

	TEST(testCudaGrid, Zoom) {
		auto dimx = 1000;
		auto dimy = 500;
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 0.01, -1, -1);
		grid.zoom(2.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 0.02, -1, -1, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCudaGrid, MultiZoom) {
		auto dimx = 1000;
		auto dimy = 500;
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.0, 0.0, 0.0);
		grid.zoom(3.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 6.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCudaGrid, OffsetZoom) {
		auto dimx = 20;
		auto dimy = 10;
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.5, 20, 10);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 2.5, -30.0, -15.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}


	TEST(testCudaGrid, Pan) {
		auto dimx = 100;
		auto dimy = 100;
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 0.1, -1, -1);
		grid.translate(-100, -200);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 0.1, 10, 20, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}


	TEST(testCudaGrid, MultiPan) {
		auto dimx = 20;
		auto dimy = 10;
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 0.1, 0.0, 0.0);
		grid.translate(-1, -2);
		grid.translate(-1, 4);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 0.1, 0.2, -0.2, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCudaGrid, Resize) {
		auto dimx = 1000;
		auto dimy = 500;
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
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
		CudaMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.resizeGrid(600, 1100);
		std::vector<RGBColor> rawGrid(600 * 1100);
		escapeTimeSequential(rawGrid, 100, 600, 1100, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < 600; i++) {
			for (int j = 0; j < 1100; j++) {
				EXPECT_EQ(rawGrid[i * 1100 + j], grid.get(i, j));
			}
		}
	}

	TEST(testCpuGrid, InitializeGrid) {
		auto dim = 5;
		CpuMandelbrotImageTransformGrid grid(dim, dim, 100, 0.1, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dim * dim);
		escapeTimeSequential(rawGrid, 100, dim, dim, 0.1, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				EXPECT_EQ(rawGrid[i * dim + j], grid.get(i, j));
			}
		}

	}

	TEST(testCpuGrid, InitializeLargeGrid) {
		auto dim = 1000;
		CpuMandelbrotImageTransformGrid grid(dim, dim, 100, 0.1, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dim * dim);
		escapeTimeSequential(rawGrid, 100, dim, dim, 0.1, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				EXPECT_EQ(rawGrid[i * dim + j], grid.get(i, j));
			}
		}

	}

	TEST(testCpuGrid, InitializeRectangleGrid) {
		auto dimx = 1000;
		auto dimy = 500;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}

	}

	TEST(testCpuGrid, InitializeWithParameters) {
		auto dimx = 1000;
		auto dimy = 500;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 4.0, 0.5, -0.5);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 4.0, 0.5, -0.5, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}

	}

	TEST(testCpuGrid, Zoom) {
		auto dimx = 1000;
		auto dimy = 500;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 2.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCpuGrid, MultiZoom) {
		auto dimx = 1000;
		auto dimy = 500;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.0, 0.0, 0.0);
		grid.zoom(3.0, 0.0, 0.0);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 6.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCpuGrid, OffsetZoom) {
		auto dimx = 20;
		auto dimy = 10;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.zoom(2.5, 20, 10);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 2.5, -30.0, -15.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}


	TEST(testCpuGrid, Pan) {
		auto dimx = 20;
		auto dimy = 10;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 0.1, 0.0, 0.0);
		grid.translate(-1, -2);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 0.1, 0.1, 0.2, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}


	TEST(testCpuGrid, MultiPan) {
		auto dimx = 20;
		auto dimy = 10;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 0.1, 0.0, 0.0);
		grid.translate(-1, -2);
		grid.translate(-1, 4);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, dimx, dimy, 0.1, 0.2, -0.2, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(rawGrid[i * dimy + j], grid.get(i, j));
			}
		}
	}

	TEST(testCpuGrid, Resize) {
		auto dimx = 1000;
		auto dimy = 500;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.resizeGrid(100, 100);
		std::vector<RGBColor> rawGrid(dimx * dimy);
		escapeTimeSequential(rawGrid, 100, 100, 100, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < 100; i++) {
			for (int j = 0; j < 100; j++) {
				EXPECT_EQ(rawGrid[i * 100 + j], grid.get(i, j));
			}
		}
	}

	TEST(testCpuGrid, ResizeLarger) {
		auto dimx = 1000;
		auto dimy = 500;
		CpuMandelbrotImageTransformGrid grid(dimx, dimy, 100, 1.0, 0.0, 0.0);
		grid.resizeGrid(600, 1100);
		std::vector<RGBColor> rawGrid(600 * 1100);
		escapeTimeSequential(rawGrid, 100, 600, 1100, 1.0, 0.0, 0.0, ColoringFunctionType::DEFAULT);

		for (int i = 0; i < 600; i++) {
			for (int j = 0; j < 1100; j++) {
				EXPECT_EQ(rawGrid[i * 1100 + j], grid.get(i, j));
			}
		}
	}


	TEST(testCudaAgainstCpu, Pan) {
		auto dimx = 20;
		auto dimy = 10;
		CudaMandelbrotImageTransformGrid cudaGrid(dimx, dimy, 100, 0.1, 0.0, 0.0);
		CpuMandelbrotImageTransformGrid cpuGrid(dimx, dimy, 100, 0.1, 0.0, 0.0);
		cudaGrid.translate(-100, -200);
		cpuGrid.translate(-100, -200);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(cudaGrid.get(i, j), cpuGrid.get(i, j));
			}
		}
	}

	TEST(testCudaAgainstCpu, PanLarge) {
		auto dimx = 2000;
		auto dimy = 1000;
		CudaMandelbrotImageTransformGrid cudaGrid(dimx, dimy, 100, 0.01, 0.0, 0.0);
		CpuMandelbrotImageTransformGrid cpuGrid(dimx, dimy, 100, 0.01, 0.0, 0.0);
		cudaGrid.translate(-100, -200);
		cpuGrid.translate(-100, -200);

		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++) {
				EXPECT_EQ(cudaGrid.get(i, j), cpuGrid.get(i, j));
			}
		}
	}
}
