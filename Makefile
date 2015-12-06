CPPFLAGS=-I/usr/local/include/opencv -I/usr/local/include -L/usr/local/cuda-6.5/lib64 /usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_ocl.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.a /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so /usr/lib/x86_64-linux-gnu/libXext.so /usr/lib/x86_64-linux-gnu/libX11.so /usr/lib/x86_64-linux-gnu/libICE.so /usr/lib/x86_64-linux-gnu/libSM.so /usr/lib/libGL.so /usr/lib/x86_64-linux-gnu/libGLU.so -lcufft -lnpps -lnppi -lnppc -lcudart -ltbb -lrt -lpthread -lm -ldl

all:	serial mpi cuda shared

serial:
	g++ -o ApplySmooth_Serial ApplySmooth_Serial.cpp $(CPPFLAGS)

mpi:	
	mpic++ -o ApplySmooth_MPI ApplySmooth_MPI.cpp $(CPPFLAGS) -fopenmp
	
shared.o:
	nvcc -c ApplySmooth_CUDA_Shared.cu
shared: shared.o
	g++ -o ApplySmooth_Shared ApplySmooth_CUDA.cpp ApplySmooth_CUDA_Shared.o $(CPPFLAGS) -fopenmp
	
cuda.o:
	nvcc -c ApplySmooth_CUDA.cu	
cuda: cuda.o
	g++ -o ApplySmooth_CUDA ApplySmooth_CUDA.cpp ApplySmooth_CUDA.o $(CPPFLAGS) -fopenmp
	
clean:
	rm ApplySmooth_Serial ApplySmooth_MPI ApplySmooth_CUDA ApplySmooth_CUDA.o
