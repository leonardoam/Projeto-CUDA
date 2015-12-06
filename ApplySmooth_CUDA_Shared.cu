/*codigo CUDA utilizando memoria compartilhada*/

#include "ApplySmooth_CUDA.h"

__global__ void smooth_cuda(unsigned short* cuda_image, unsigned short* new_cuda_image, int rows, int cols){
	__shared__ short localcopy[12][8];

	int coordX = (blockIdx.x*blockDim.x)+threadIdx.x;
	int coordY = (blockIdx.y*blockDim.y)+threadIdx.y;

	int baseX = blockIdx.x*blockDim.x-2;
	int baseY = blockIdx.y*blockDim.y-2;

	int localX = threadIdx.x+2;
	int localY = threadIdx.y+2;

	/*copia da memoria global para a compartilhada*/
	if (threadIdx.x < 6){
		for(int i=threadIdx.x*2; i<=threadIdx.x*2+1; i++){
			for(int j=threadIdx.y*2; j<=threadIdx.y*2+1; j++){
				int x = baseX+i;
				int y = baseY+j;
				if (x >= 0 && y >= 0 && x < rows*cols && y < rows*cols){
					localcopy[i][j] = cuda_image[coord(x,y)];
				}else{
					localcopy[i][j] = -1;
				}
			}
		}
	}
	__syncthreads();
	unsigned int sum = 0;
	int count = 0;
	for(int i=localX-2; i<=localX+2; i++){
		for(int j=localY-2; j<=localY+2; j++){
			if (localcopy[i][j] != -1){
				sum += localcopy[i][j];
				count++;
			}
		}
	}

	if (count > 0)
		new_cuda_image[coord(coordX, coordY)] = sum/count;
	//printf("%d\t%d\n", cuda_image[coord(coordX, coordY)], new_cuda_image[coord(coordX, coordY)]);
}

void smooth(unsigned short *image, int rows, int cols){
	unsigned short* cuda_image;
	unsigned short* new_cuda_image;

	cudaMalloc(&cuda_image, rows*cols*sizeof(unsigned short));
	cudaMalloc(&new_cuda_image, rows*cols*sizeof(unsigned short));
	cudaMemcpy(cuda_image, image, rows*cols*sizeof(unsigned short), cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(8,4);
	dim3 numBlocks(rows/threadsPerBlock.x, cols/threadsPerBlock.y);
	//dim3 numBlocks(1,1);
	smooth_cuda<<<numBlocks, threadsPerBlock>>>(cuda_image, new_cuda_image, rows, cols);
	cudaDeviceSynchronize();
	cudaError_t cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		printf("Cuda Error: %s\n", cudaGetErrorString(cuda_error));

	cudaMemcpy(image, new_cuda_image, rows*cols*sizeof(unsigned short), cudaMemcpyDeviceToHost);
	
	cudaFree(cuda_image);
	cudaFree(new_cuda_image);
	return;
}
