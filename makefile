# modify the following 6 lines
nvcc=/home/cuda-10.0/bin/nvcc
cuda_inc=/home/cuda-10.0/include/
cuda_lib=/home/cuda-10.0/lib64/
nsync=/home/anaconda3/envs/dev/lib/python3.6/site-packages/tensorflow/include/external/nsync/public
tf_inc = /home/anaconda3/envs/dev/lib/python3.6/site-packages/tensorflow_core/include
tf_lib = /home/anaconda3/envs/dev/lib/python3.6/site-packages/tensorflow_core

all: tf_emd_so.so

tf_emd.cu.o: tf_emd.cu
	$(nvcc) tf_emd.cu -o tf_emd.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_emd_so.so: tf_emd.cpp tf_emd.cu.o
	g++ -shared $(CPPFLAGS) tf_emd.cpp tf_emd.cu.o -o tf_emd_so.so \
	-I $(cuda_inc) -I $(tf_inc) \
	-L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

clean:
	rm -rf *.o *.so
