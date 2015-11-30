#include <iostream>
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace std;

void smooth(Mat &image){
	/*nova matriz BGR (padrao opencv)*/
	vector<vector<vector<unsigned short> > > *newimage = new vector<vector<vector<unsigned short> > >(image.rows, vector<vector<unsigned short> >(image.cols, vector<unsigned short>(3)));
	/*smooth*/
	for(int i=0; i<image.rows; i++){
		for(int j=0; j<image.cols; j++){
			/*calcula media de cada componente BGR*/
			unsigned int b = 0;
			unsigned int g = 0;
			unsigned int r = 0;
			int count = 0;
			for(int k=max(i-2, 0); k<=min(i+2, image.rows-1); k++){
				for(int l=max(j-2, 0); l<=min(j+2, image.cols-1); l++){
					b += image.data[image.step[0]*k + image.step[1]* l + 0];
					g += image.data[image.step[0]*k + image.step[1]* l + 1];	
					r += image.data[image.step[0]*k + image.step[1]* l + 2];
					count++;
				}
			}
			if (count < 1){
				continue;
			}
			b /= count;
			g /= count;
			r /= count;

			/*escreve em nova matriz*/
			newimage->at(i).at(j).at(0) = (unsigned short) b;
			newimage->at(i).at(j).at(1) = (unsigned short) g;
			newimage->at(i).at(j).at(2) = (unsigned short) r;
		}
	}

	/*atualiza imagem de entrada*/
	for(int i=0; i<image.rows; i++){
		for(int j=0; j<image.cols; j++){
           		*(&(image.data[image.step[0]*i + image.step[1]* j + 0])) = newimage->at(i).at(j).at(0);
		        *(&(image.data[image.step[0]*i + image.step[1]* j + 1])) = newimage->at(i).at(j).at(1);
           		*(&(image.data[image.step[0]*i + image.step[1]* j + 2])) = newimage->at(i).at(j).at(2);
		}
	}
	delete newimage;
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

	/*aplica smooth*/
	smooth(image);
	//cout << endl << endl << endl;

	/*gera nome do arquivo de saida (entrada.ext -> entrada_smooth.ext)*/
	string oldname(argv[1]);
	size_t pos = oldname.rfind('.');
	string newname;
	newname.append(oldname.substr(0, pos));
	newname.append("_smooth");
	newname.append(oldname.substr(pos, oldname.size()-pos));

	/*salva imagem processada*/
	int res = imwrite(newname.c_str(), image);
	if (!res){
		cout << "Erro ao criar nova imagem.\n";
		cout << cvErrorStr(res) << endl;
	}else;
		//cout << "Nova imagem criada com sucesso.\n";
	return 0;
}
