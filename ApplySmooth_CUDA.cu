#include "ApplySmooth_CUDA.h"

__global__ void smooth_cuda(unsigned short* cuda_image, unsigned short* new_cuda_image, int rows, int cols){
	int coordX = (blockIdx.x*blockDim.x)+threadIdx.x;
	int coordY = (blockIdx.y*blockDim.y)+threadIdx.y;

	int n = rows*cols;
	int count = 0;
	unsigned int sum = 0;
	for(int i=max(0, coordX-2); i<=min(n-1, coordX+2); i++){
		for(int j=max(0, coordY-2); j<=min(n-1, coordY+2); j++){
			sum += cuda_image[coord(i, j)];
			count++;
		}
	}
	if (count > 0)
		new_cuda_image[coord(coordX, coordY)] = sum/count;
}

void smooth(unsigned short *image, int rows, int cols){
	unsigned short* cuda_image;
	unsigned short* new_cuda_image;

	cudaMalloc(&cuda_image, rows*cols*sizeof(unsigned short));
	cudaMalloc(&new_cuda_image, rows*cols*sizeof(unsigned short));
	cudaMemcpy(cuda_image, image, rows*cols*sizeof(unsigned short), cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(8,4);
	dim3 numBlocks(rows/threadsPerBlock.x, cols/threadsPerBlock.y);
	smooth_cuda<<<numBlocks, threadsPerBlock>>>(cuda_image, new_cuda_image, rows, cols);

	cudaError_t cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		printf("Cuda Error: %s\n", cudaGetErrorString(cuda_error));

	cudaMemcpy(image, new_cuda_image, rows*cols*sizeof(unsigned short), cudaMemcpyDeviceToHost);
	
	cudaFree(cuda_image);
	cudaFree(new_cuda_image);
	return;
}
