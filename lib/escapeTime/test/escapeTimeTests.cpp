#include <gtest/gtest.h>
#include <vector>
#include <iostream>

#include "escapeTimeSequential.h"
#include "escapeTimeCudaWrapper.h"

namespace cudaEscapeTime {
	TEST(testCudaEscapeTime, OutOfRadius) {
		RGBColor escapeTime;
		RGBColor* escapeTimePtr = &escapeTime;
		escapeTimeCUDA(escapeTimePtr, 1000, 1, 1, 1.0, 4.0, 4.0);
		ASSERT_EQ(escapeTime, 1);
	}

	TEST(testCudaEscapeTime, EscapesAfterMultipleIterations) {
		RGBColor escapeTime;
		RGBColor* escapeTimePtr = &escapeTime;
		escapeTimeCUDA(escapeTimePtr, 1000, 1, 1, 1.0, -0.0318, -0.8614);
		ASSERT_EQ(escapeTime, 11);
	}

	TEST(testCudaEscapeTime, DoesNotEscape) {
		RGBColor escapeTime;
		RGBColor* escapeTimePtr = &escapeTime;
		escapeTimeCUDA(escapeTimePtr, 256, 1, 1, 1.0, -0.2583, 0.6562);
		ASSERT_EQ(escapeTime, 256);
	}
	
	/*
	* Some important cases for the CUDA escape time algorithm involve large grids that are impractical to handwrite
	* test cases for. Instead, after verifying the sequential escape time algorithm, the test cases for the CUDA algorithm
	* verify against the sequential algorithm. 
	*/
	TEST(testCudaEscapeTime, SmallGrid) {
		int sizeX = 16;
		int sizeY = 16;
		int iters = 50;
		double scale = 0.1;
		double panX = -1;
		double panY = -1;

		std::vector<RGBColor> seqGrid(sizeX * sizeY);
		std::vector<RGBColor> cudaGrid(sizeX * sizeY);
		escapeTimeCUDA(cudaGrid.data(), iters, sizeX, sizeY, scale, panX, panY);
		escapeTimeSequential(seqGrid, iters, sizeX, sizeY, scale, panX, panY, ColoringFunction::Function::DEFAULT);

		ASSERT_EQ(cudaGrid, seqGrid);
	}

	TEST(testCudaEscapeTime, LargeGrid) {
		int sizeX = 256;
		int sizeY = 256;
		int iters = 50;
		double scale = 0.01;
		double panX = -1;
		double panY = -1;

		std::vector<RGBColor> seqGrid(sizeX * sizeY);
		std::vector<RGBColor> cudaGrid(sizeX * sizeY);
		escapeTimeCUDA(cudaGrid.data(), iters, sizeX, sizeY, scale, panX, panY);
		escapeTimeSequential(seqGrid, iters, sizeX, sizeY, scale, panX, panY, ColoringFunction::Function::DEFAULT);
		ASSERT_EQ(cudaGrid, seqGrid);
	}

	TEST(testCudaEscapeTime, UnevenThreadBlocks) {
		int sizeX = 250;
		int sizeY = 120;
		int iters = 50;
		double scale = 0.01;
		double panX = -1;
		double panY = -1;

		std::vector<RGBColor> seqGrid(sizeX * sizeY);
		std::vector<RGBColor> cudaGrid(sizeX * sizeY);
		escapeTimeCUDA(cudaGrid.data(), iters, sizeX, sizeY, scale, panX, panY);
		escapeTimeSequential(seqGrid, iters, sizeX, sizeY, scale, panX, panY, ColoringFunction::Function::DEFAULT);

		ASSERT_EQ(cudaGrid, seqGrid);
	}

	TEST(testCudaEscapeTime, LargerXThanY) {
		int sizeX = 255;
		int sizeY = 127;
		int iters = 50;
		double scale = 0.01;
		double panX = -1;
		double panY = -1;

		std::vector<RGBColor> cudaGrid(sizeX * sizeY);
		std::vector<RGBColor> seqGrid(sizeX * sizeY);
		escapeTimeCUDA(cudaGrid.data(), iters, sizeX, sizeY, scale, panX, panY);
		escapeTimeSequential(seqGrid, iters, sizeX, sizeY, scale, panX, panY, ColoringFunction::Function::DEFAULT);
		ASSERT_EQ(cudaGrid, seqGrid);
	}

	TEST(testCudaEscapeTime, UnevenDims) {
		int sizeX = 3;
		int sizeY = 4;
		int iters = 50;
		double scale = 0.01;
		double panX = -1;
		double panY = -1;

		std::vector<RGBColor> cudaGrid(sizeX * sizeY);
		std::vector<RGBColor> seqGrid(sizeX * sizeY);
		escapeTimeCUDA(cudaGrid.data(), iters, sizeX, sizeY, scale, panX, panY);
		escapeTimeSequential(seqGrid, iters, sizeX, sizeY, scale, panX, panY, ColoringFunction::Function::DEFAULT);
		ASSERT_EQ(cudaGrid, seqGrid);
	}
}

namespace sequentialEscapeTime {
	TEST(testPointEscapeTime, OutOfRadius) {
		int escapeTime = pointEscapeTime(4, 4, 256);
		ASSERT_EQ(escapeTime, 1);
	}

	TEST(testPointEscapeTime, EscapesAfterMultipleIterations) {
		int escapeTime = pointEscapeTime(-0.0318, -0.8614, 256);
		ASSERT_EQ(escapeTime, 11);
	}

	TEST(testPointEscapeTime, DoesNotEscape) {
		int escapeTime = pointEscapeTime(-0.2583, 0.6562, 256);
		ASSERT_EQ(escapeTime, 256);
	}

	TEST(testSequentialEscapeTime, SquareGridWithScaleAndPan) {
		std::vector<RGBColor> grid(9);
		escapeTimeSequential(grid, 10, 3, 3, 0.3, 0.2, 0.1, ColoringFunction::Function::DEFAULT);
		double xPoints[3] = {0.2, 0.5, 0.8};
		double yPoints[3] = { 0.1, 0.4, 0.7 };
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				ASSERT_EQ(grid[i*3 + j], pointEscapeTime(xPoints[i], yPoints[j], 10));
			}
		}
		
	}

	TEST(testSequentialEscapeTime, LargerXThanY) {
		std::vector<RGBColor> grid(6);
		escapeTimeSequential(grid, 10, 3, 2, 0.3, 0.2, 0.1, ColoringFunction::Function::DEFAULT);
		double xPoints[3] = { 0.2, 0.5, 0.8 };
		double yPoints[2] = { 0.1, 0.4 };
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				ASSERT_EQ(grid[i * 2 + j], pointEscapeTime(xPoints[i], yPoints[j], 10));
			}
		}

	}

}
