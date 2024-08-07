__global__ void escapeTime(int** escapeTimes, float scale, float panX, float panY) {
	float real = threadIdx.x * scale + panX;
	float imaginary = threadIdx.y + panY;


}