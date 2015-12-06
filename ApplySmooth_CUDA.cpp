/*arquivo responsavel pela manipulacao das imagens*/

#include <bits/stdc++.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "ApplySmooth_CUDA.h"

using namespace cv;
using namespace std;

int cols;
void smooth(Mat &image){
	/*alocacao de memoria e leitura*/
	unsigned short *cuda_imageB = (unsigned short*) malloc (image.rows*image.cols*sizeof(unsigned short));
	unsigned short *cuda_imageG = (unsigned short*) malloc (image.rows*image.cols*sizeof(unsigned short));
	unsigned short *cuda_imageR = (unsigned short*) malloc (image.rows*image.cols*sizeof(unsigned short));

	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic) nowait 
		for(int i=0; i<image.rows; i++){
			for(int j=0; j<image.cols; j++){
				cuda_imageB[coord(i, j)] = image.data[image.step[0]*i + image.step[1]*j + 0];
				cuda_imageG[coord(i, j)] = image.data[image.step[0]*i + image.step[1]*j + 1];
				cuda_imageR[coord(i, j)] = image.data[image.step[0]*i + image.step[1]*j + 2];
			}
		}
	}

	/*chamada do codigo CUDA linkado*/
	smooth(cuda_imageB, image.rows, image.cols);
	smooth(cuda_imageG, image.rows, image.cols);
	smooth(cuda_imageR, image.rows, image.cols);

	/*saida e liberacao de memoria*/
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic) nowait 
		for(int i=0; i<image.rows; i++){
			for(int j=0; j<image.cols; j++){
				image.data[image.step[0]*i+image.step[1]*j+0] = cuda_imageB[coord(i, j)];
				image.data[image.step[0]*i+image.step[1]*j+1] = cuda_imageG[coord(i, j)];
				image.data[image.step[0]*i+image.step[1]*j+2] = cuda_imageR[coord(i, j)];
			}
		}
	}

	free(cuda_imageB);
	free(cuda_imageG);
	free(cuda_imageR);
}

int main(int argc, char** argv ){
//	ios::sync_with_stdio(0);
	if ( argc != 2 ){
		cout << "usage: DisplayImage.out <Image_Path>\n";
		return -1;
	}

	/*le arquivo*/
	Mat temp = imread(argv[1], CV_LOAD_IMAGE_COLOR);	
	Mat image;
	temp.convertTo(image, CV_16UC3);

	if (!image.data){
		cout << "No image data\n";
		return -1;
	}

	cols = image.cols; /*nao mexa e evite usar outra variavel com esse nome*/

	/*aplica smooth*/
	smooth(image);

	/*gera nome do arquivo de saida (entrada.ext -> entrada_smooth.ext)*/
	string oldname(argv[1]);
	size_t pos = oldname.rfind('.');
	string newname;
	newname.append(oldname.substr(0, pos));
	newname.append("_smooth.jpg");

	/*salva imagem processada*/
	int res = imwrite(newname.c_str(), image);
	if (!res){
		cout << "Erro ao criar nova imagem.\n";
		cout << cvErrorStr(res) << endl;
	}else;
		//cout << "Nova imagem criada com sucesso.\n";
	return 0;
}
