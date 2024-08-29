#include <chrono>
#include <stdexcept>
#include <vector>
#include <iostream>

#include "escapeTime/escapeTimeCuda.h"
#include "escapeTime/escapeTimeSequential.h"

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Must have at least 2 cmd arguments: sizeX and sizeY";
		return 1;
	}

	int sizeX = atoi(argv[1]);
	int sizeY = atoi(argv[2]);

	int iters = 50;

	if (sizeX <= 0 || sizeY <= 0) {
		std::cerr << "sizeX and sizeY must be positive integers";
		return 2;
	}

	if (argc > 3) {
		iters = atoi(argv[3]);
		if (iters <= 0) {
			std::cerr << "iters must be a positive integer";
			return 2;
		}
	}


	std::vector<int> cudaGrid(sizeX * sizeY);
	std::vector<int> seqGrid(sizeX * sizeY);

	float step = 2 / (float)sizeX;

	auto start = std::chrono::steady_clock::now();
	escapeTimeCUDA(cudaGrid.data(), iters, sizeX, sizeY, step, -1, -1);
	auto end = std::chrono::steady_clock::now();
	auto cudaTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	start = std::chrono::steady_clock::now();
	escapeTimeSequential(seqGrid.data(), iters, sizeX, sizeY, step,-1, -1);
	end = std::chrono::steady_clock::now();
	auto seqTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "CUDA elapsed time: " << cudaTime.count() << "ms\n";
	std::cout << "Sequential elapsed time: " << seqTime.count() << "ms\n";

}