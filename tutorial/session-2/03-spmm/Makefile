CUDA_INC = /usr/local/cuda/include/
CUDA_LIB = /usr/local/cuda/lib64/


all: libbsr.so


libbsr.so : bsr.cpp 
	${CXX} -O3 --shared -fPIC -I${CUDA_INC} $<  -o $@ -lcudart -lcusparse -L${CUDA_LIB} -Wl,-rpath,${CUDA_LIB} 