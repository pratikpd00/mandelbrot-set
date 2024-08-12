__global__ void escapeTime(int** escapeTimes, int max_iters, double scale, double panX, double panY) {
	//Mandelbrot recurrence for reference: z_n+1 = z_n^2 + c
	const double cReal = threadIdx.x * scale + panX;
	const double cImaginary = threadIdx.y + panY;

	double zReal = 0;
	double zImaginary = 0;

	double zRealSqr = 0;
	double zImaginarySqr = 0;

	//if the absolute value of z ever exceeds 2, it is guaranteed to escape the mandelbrot set
	int i;
	for (i = 0; i < max_iters && zRealSqr + zImaginarySqr <= 4; i++) {
		//(ai + b) ^2 = -a^2 + 2abi + b^2
		zImaginary = 2*zReal*zImaginary + cImaginary;
		zReal = zRealSqr - zImaginarySqr + cReal;
		
		zRealSqr = zReal*zReal;
		zImaginarySqr = zImaginary*zImaginary;
	}

	escapeTimes[threadIdx.x][threadIdx.y] = i; 
}

void CudaEscapeTime() {

}