__global__ void escapeTime(int max_iters, float scale, float panX, float panY, int** escapeTimes) {
	//Mandelbrot recurrence for reference: z_n+1 = z_n^2 + c
	float cReal = threadIdx.x * scale + panX;
	float cImaginary = threadIdx.y + panY;

	float zReal = 0;
	float zImaginary = 0;

	for (int i = 0; i < max_iters && zReal ^ 2 + zImaginary ^ 2 <= 4; i++) {

	}


}