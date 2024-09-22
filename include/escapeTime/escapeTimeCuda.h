#ifndef ESCAPE_TIME_CUDA_PUBLIC
#define ESCAPE_TIME_CUDA_PUBLIC

void escapeTimeCUDA(int* escapeTimes, int maxIters, int sizeX, int sizeY, double scale, double panX, double panY);

#ifdef __CUDACC__
__global__ void escapeTimeKernel(int* escapeTimes, int max_iters, int sizeX, int sizeY, double scale, double panX, double panY);
#endif // 


#endif 
