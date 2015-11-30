#include <bits/stdc++.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "mpi.h"
#include <string.h>
#include <cmath>

using namespace cv;
using namespace std;

void smooth(unsigned short** image, int startr, int endr, int startc, int endc){
	/*nova matriz BGR (padrao opencv)*/
	vector<vector<vector<unsigned short> > > *newimage = new vector<vector<vector<unsigned short> > >(endr, vector<vector<unsigned short> >(endc, vector<unsigned short>(3)));
	int tid; unsigned int value;
	/*smooth*/
	#pragma omp parallel num_threads(3) shared(newimage, image) private(tid, value) firstprivate(startr, endr, startc, endc)
	{
		tid = omp_get_thread_num();		
		//printf("TID: %d\n", tid);
		for(int i=startr; i<endr; i++){
			for(int j=startc; j<endc; j++){
				/*calcula media de cada componente BGR*/
				value = 0;
				int count = 0;
				for(int k=max(i-2, 0); k<=min(i+2, endr-1); k++){
					for(int l=max(j-2, 0); l<=min(j+2, endc-1); l++){
						value += image[k][l*3+tid];
						count++;
					}
				}
				value /= count;
				
				/*escreve em nova matriz*/
				newimage->at(i).at(j).at(tid) = (unsigned short) value;
			}
		}
	}
	
	#pragma omp parallel shared(image, newimage) firstprivate(startr, endr, startc, endc)
	{
		#pragma omp for schedule(dynamic) nowait 
		for(int i=startr; i<endr; i++)
			for(int j=startc; j<endc; j++){
				*(&(image[i][j*3+0])) = newimage->at(i).at(j).at(0);
				*(&(image[i][j*3+1])) = newimage->at(i).at(j).at(1);
				*(&(image[i][j*3+2])) = newimage->at(i).at(j).at(2);
			}
	}

	delete(newimage);
}

void printImage(Mat &image, int starts[], int n){
	for(int i = 1; i < n; i++){
		printf("\n\n\n\nsent to rank %d, numero de colunas: %d\n\n", i,((i < n-1)?starts[i+1] : image.cols)-starts[i]);
		for(int j = 0; j < image.rows; j++){
			for(int k = starts[i]; k < ((i < n-1)?starts[i+1] : image.cols); k++){
				printf("%d %d %d\n", image.data[image.step[0]*j + image.step[1]* k + 0], image.data[image.step[0]*j + image.step[1]* k + 1], image.data[image.step[0]*j + image.step[1]* k + 2]);	
			}
		}

	}
}

void printLocalImage(unsigned short** image, int beginExtra, int localCols, int rank, int nrows){	
	printf("\n\n\n\nreceived on rank %d, localCols %d, beginExtra %d\n\n", rank, localCols, beginExtra);
	for(int j = 0; j < nrows; j++){
		for(int k = beginExtra; k < beginExtra+localCols; k++){
			printf("%d %d %d\n", image[j][k*3], image[j][k*3+1], image[j][k*3+2]);	
		}
	}
}

void updateImage(unsigned short **newimage, Mat &image){
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic) nowait 
		for(int i=0; i<image.rows; i++){
			for(int j=0; j<image.cols; j++){
//				printf("master %d %d %d\n", newimage[i][j*3+0], newimage[i][j*3+1], newimage[i][j*3+2]);
				image.data[image.step[0]*i + image.step[1]* j + 0] = newimage[i][j*3+0];
				image.data[image.step[0]*i + image.step[1]* j + 1] = newimage[i][j*3+1];
				image.data[image.step[0]*i + image.step[1]* j + 2] = newimage[i][j*3+2];
			}
			free(newimage[i]);
		}
	}

	free(newimage);
}

int main(int argc, char* argv[]){
	if ( argc != 2 ){
		cout << "usage: DisplayImage.out <Image_Path>\n";
		return -1;
	}

	//cout << "starting Apply Smooth MPI\n";
	/*MPI initialize*/
	MPI_Init(&argc, &argv);

	int n, rank, nrows, ncols, localCols, localStartCol, localStartRow, localEndCol, localEndRow;
	MPI_Comm_size(MPI_COMM_WORLD, &n);

	if(n < 2){
		printf("Deve-se utilizar ao menos 2 processos!\n");

		MPI_Finalize();
		return 0;
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	

	//printf("starting: %d\n", rank);
	Mat image;
	//master only
	if(rank == 0){
		/*read file*/
		Mat temp = imread(argv[1], CV_LOAD_IMAGE_COLOR);
		temp.convertTo(image, CV_16UC3);

		if (!image.data){
			cout << "No image data\n";
			return -1;
		}
	
		nrows = image.rows;
		ncols = image.cols;
	}

	//let everyone knows the number of rows and columns 
	MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//if(rank == 0)
	//	printf("nrows: %d, ncols: %d\n", nrows, ncols);

	//calculate the number of columns each process will take and its position in the Map
	localCols = ncols/(n-1);
	if(localCols < 1){
		if(rank ==  0)
			printf("Numero de processos muito grande!\n");
		MPI_Finalize();
		return 1;	
	}

	if(rank > 0){
		if(rank <= ncols%(n-1))
			localCols++;

	}

	localStartRow = 0; localEndRow = nrows-1;
	localStartCol = (rank-1)*(ncols/(n-1))+((rank-1 < ncols%(n-1) && ncols%(n-1) > 0)? rank-1 : ncols%(n-1));
	localEndCol = localStartCol+localCols;

	//everybody will keep track of each one's starting column and number of columns 
	int starts[n];
	int displs[n];
	int sizes[n];	
	if(rank == 0){
		localStartCol = 0;
		localCols = 0;
		localEndCol = ncols-1;
		localStartRow = 0;
		localEndRow = nrows-1;
	}
	int localSize = localCols*3;	
	MPI_Allgather(&localStartCol, 1, MPI_INT, starts, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(&localSize, 1, MPI_INT, sizes, 1, MPI_INT, MPI_COMM_WORLD);

	if(rank == 0){
		for(int i = 0; i < n; i++){
			//printf("starts[%d]: %d\n", i, starts[i]);
			//printf("sizes[%d]: %d\n", i, sizes[i]);
			displs[i] = 3*starts[i];
		}
	}

	int tag;
	char aux[20];
	sprintf(aux, "%d", ncols);
	int auxLen = strlen(aux);
	int p = (int)pow(10, auxLen); 
	int size = sizeof(unsigned short)*3;
	int retval;

	//master only
	if(rank == 0){
		#pragma omp parallel private(tag, size, retval)
		{
			//printf("master gonna send data\n");
			#pragma omp for schedule(dynamic) nowait 
			for(int i = 1; i < n; i++){
				int beginExtra = (i == 1)? 0 : 2;
				int endExtra = ((i == n-1)? 0 : 2);
				int totalCols = sizes[i]/3+beginExtra+endExtra;
				
				unsigned short* outbuf = (unsigned short*) malloc(sizeof(unsigned short)*totalCols*3);
			//	printf("from master, i: %d, starts[i]: %d\n", i, starts[i]);
				for(int k = 0; k < nrows; k++){
					tag = k;
					for(int j = starts[i]; j < ((i+1 < n)? starts[i+1] : ncols); j++){			
						//printf("from master, to: %d, k: %d, j: %d\n", i, k, j);
						//appending to beggining
						if(i > 1 && j-starts[i] == 0){	
							outbuf[0] = image.data[image.step[0]*k + image.step[1]*(j-2) + 0];
							outbuf[1] = image.data[image.step[0]*k + image.step[1]*(j-2) + 1];
							outbuf[2] = image.data[image.step[0]*k + image.step[1]*(j-2) + 2];
	
							outbuf[3] = image.data[image.step[0]*k + image.step[1]*(j-1) + 0];
							outbuf[4] = image.data[image.step[0]*k + image.step[1]*(j-1) + 1];
							outbuf[5] = image.data[image.step[0]*k + image.step[1]*(j-1) + 2];
						}
						
						else if(i < n-1 && starts[i+1]-j == 1){	//appending to end
							outbuf[(totalCols-2)*3] = image.data[image.step[0]*k + image.step[1]*(j+1) + 0];
							outbuf[(totalCols-2)*3+1] = image.data[image.step[0]*k + image.step[1]*(j+1) + 1];
							outbuf[(totalCols-2)*3+2] = image.data[image.step[0]*k + image.step[1]*(j+1) + 2];
	
							outbuf[(totalCols-1)*3] = image.data[image.step[0]*k + image.step[1]*(j+2) + 0];
							outbuf[(totalCols-1)*3+1] = image.data[image.step[0]*k + image.step[1]*(j+2) + 1];
							outbuf[(totalCols-1)*3+2] = image.data[image.step[0]*k + image.step[1]*(j+2) + 2];
						}
						
						
						outbuf[(j-starts[i]+beginExtra)*3+0] = image.data[image.step[0]*k + image.step[1]*j + 0];
						outbuf[(j-starts[i]+beginExtra)*3+1] = image.data[image.step[0]*k + image.step[1]*j + 1];
						outbuf[(j-starts[i]+beginExtra)*3+2] = image.data[image.step[0]*k + image.step[1]*j + 2];
					}
					//send pixel data to the i-th process and to neighbor where necessary (for bordering pixels)
					#pragma omp critical
					{
						retval = MPI_Send(outbuf, 3*totalCols, MPI_UNSIGNED_SHORT, i, tag, MPI_COMM_WORLD);
					}
				}
				free(outbuf);
			}
			//printf("master finished sending data: %d\n", rank);
		}
	}
	 
	MPI_Status* stat = (MPI_Status*) malloc(sizeof(MPI_Status));
	int beginExtra = (localStartCol == 0)? 0 : 2;
	int endExtra = ((rank == n-1)? 0 : 2);
	unsigned short** localImage; 
	
	int localTotalCols = 0;
	
	if(rank > 0){
		localImage = (unsigned short**) malloc(sizeof(unsigned short*)*nrows);
		localTotalCols = localCols+beginExtra+endExtra;
		unsigned short* recvbuf = (unsigned short*) malloc(sizeof(unsigned short)*3*localTotalCols);
	//	printf("gonna receive data on rank: %d, nrows: %d, ncols: %d, init k: %d, end k: %d, init j: %d, end j: %d, localTotalCols: %d\n", rank, nrows, ncols, 0, nrows, 0, localTotalCols, localTotalCols);
		for(int k = 0; k < nrows; k++){
			tag = k;
			localImage[k] = (unsigned short*) malloc(sizeof(unsigned short)*3*localTotalCols);

			MPI_Recv(recvbuf, 3*localTotalCols, MPI_UNSIGNED_SHORT, 0, tag, MPI_COMM_WORLD, stat);
				
			for(int j = 0; j < localTotalCols; j++){ 	

				
				localImage[k][j*3] = recvbuf[j*3+0];
				localImage[k][j*3+1] = recvbuf[j*3+1];
				localImage[k][j*3+2] = recvbuf[j*3+2];
		//		if(rank == 1){	
		//			printf("recvtag: %d\n", tag);
	//				printf("master %d %d %d\n", localImage[k][j*3], localImage[k][j*3+1], localImage[k][j*3+2]);
		//		}
			}
		}
		free(recvbuf);
			
		//printf("finished receiving data: %d\n", rank);

		/*apply smooth*/
		//printf("aplicando smooth em parte %d\n", rank);
		smooth(localImage, 0, nrows, 0, localTotalCols);
	}

	/*int a;
	if(rank > 0)
              MPI_Recv(&a, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, stat);

	if(rank > 0)
        	;//printLocalImage(localImage, beginExtra, localCols, rank, nrows);
	else 
		;//printImage(image, starts, n);	
		
        if(rank < n-1)
              MPI_Send(&a, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
	*/

	free(stat);
	unsigned short** newimage;
	unsigned short* sendbuf;
	unsigned short* recvbuf;

	if(rank == 0)
		newimage = (unsigned short**) malloc(nrows*sizeof(unsigned short*));

	for(int i = localStartRow; i <= localEndRow; i++){
		if(rank > 0){
			sendbuf = (unsigned short*) malloc(sizeof(unsigned short)*localCols*3);
			recvbuf = (unsigned short*) malloc(sizeof(unsigned short));
			for(int j = beginExtra; j < beginExtra+localCols; j++){
				sendbuf[(j-beginExtra)*3] = localImage[i][j*3];
				sendbuf[(j-beginExtra)*3+1] = localImage[i][j*3+1];
				sendbuf[(j-beginExtra)*3+2] = localImage[i][j*3+2];
				//printf("rank: %d, %d, %d, %d\n", rank, sendbuf[(j-beginExtra)*3], sendbuf[(j-beginExtra)*3+1], sendbuf[(j-beginExtra)*3+2]);
			}
				
		}else{
			newimage[i] = (unsigned short*) malloc(ncols*3*sizeof(unsigned short));
			sendbuf = (unsigned short*) malloc(sizeof(unsigned short));	
			recvbuf = (unsigned short*) malloc(sizeof(unsigned short)*ncols*3);
		}
		//printf("gonna do gatherv of row %d on %d, sizes[rank]: %d == 3*localCols: %d\n", i, rank, sizes[rank], 3*localCols);
		MPI_Gatherv(sendbuf, 3*localCols, MPI_UNSIGNED_SHORT, recvbuf, sizes, displs, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);	
		if(rank == 0){
			for(int j = 0; j < ncols*3; j++)
				newimage[i][j] = recvbuf[j];
		}
		free(sendbuf);
		free(recvbuf);
	}
	
	if(rank > 0){
		for(int i = 0; i < nrows; i++)
			free(localImage[i]);
		free(localImage);
	}

	//master only
	if(rank == 0){
		//udate the Map using the calculated values
		updateImage(newimage, image);

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

	}

	/*MPI finalize*/
	MPI_Finalize();

	return 0;
}
