#ifndef ESCAPE_TIME_CUDA_PUBLIC
#define ESCAPE_TIME_CUDA_PUBLIC

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

	void escapeTimeCUDA(int* escapeTimes, int max_iters, int sizeX, int sizeY, double scale, double panX, double panY);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif 
