FROM ubuntu:20.04 AS base

LABEL maintainer="rizwan.ashraf@pnnl.gov"

RUN apt-get update && \
	apt-get -y --no-install-recommends install \
    build-essential \
    vim \
	xz-utils \
	ca-certificates \
	git \
	wget \
	ninja-build \
	python3 \
    curl && \
	apt-get clean

# get the cmake v3.27
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.3/cmake-3.27.3-linux-x86_64.sh \
	-q -O /tmp/cmake-install.sh \
	&& chmod u+x /tmp/cmake-install.sh \
	&& mkdir /opt/cmake-3.27.3 \
	&& /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.27.3 \
	&& rm /tmp/cmake-install.sh \
	&& ln -s /opt/cmake-3.27.3/bin/* /usr/local/bin

FROM base AS build-graphblas

WORKDIR /src

# build graphBLAS
RUN git clone --depth 1 https://github.com/DrTimothyAldenDavis/GraphBLAS.git -b v7.3.2 && \
    cd GraphBLAS && \
	cd build && \
	cmake -G Ninja .. && \
	ninja && \
	cd ../..

# build LAGraph (graphBLAS is a dependency)
RUN git clone --depth 1 https://github.com/johnpzh/LAGraph.git -b master && \
	cd LAGraph && \
    cd build && \
    cmake .. && \
	make JOBS=16 


FROM base AS build-comet

WORKDIR /src

# git clone
RUN git clone --depth 1 https://github.com/pnnl/COMET.git -b pact23-ae && \
    cd COMET && \
    git submodule init && \
    git submodule update 

# build llvm
RUN	cd /src/COMET && \
    mkdir llvm/build && \
	cd llvm/build && \
	cmake -G Ninja ../llvm \
         -DLLVM_ENABLE_PROJECTS="mlir" \
         -DLLVM_TARGETS_TO_BUILD="X86" \
         -DLLVM_ENABLE_ASSERTIONS=ON \
         -DCMAKE_BUILD_TYPE=Release && \
    ninja 

# build blis (COMET dependency)
RUN cd /src/COMET && \
    patch -s -p0 < comet-blis.patch && \
    mkdir install && \
    cd blis && \
    ./configure --prefix=/src/COMET/install auto && \
    make -j16 && \
    make install
    
# build COMET 
RUN cd /src/COMET && \
    git pull && \
    mkdir build && cd build && \
    cmake -G Ninja .. \
         -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
         -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
         -DLLVM_ENABLE_ASSERTIONS=ON \
         -DCMAKE_BUILD_TYPE=Release && \
    ninja

FROM ubuntu:20.04

RUN apt-get update && \
        apt-get -y --no-install-recommends install \
        python3 \
        python3-pip \
        curl \
        vim \
        libgomp1 && \
        apt-get clean

# get graphBLAS lib
COPY --from=build-graphblas /src/GraphBLAS/build/libgraphblas.so.7.3.2 /lib/graphblas/
COPY --from=build-graphblas /src/GraphBLAS/build/libgraphblas.so.7 /lib/graphblas/
COPY --from=build-graphblas /src/GraphBLAS/build/libgraphblas.so /lib/graphblas/

# get LAGraph lib
COPY --from=build-graphblas /src/LAGraph/build/liblagraph.a /lib/graphblas/
COPY --from=build-graphblas /src/LAGraph/build/liblagraph.so.1.0.1 /lib/graphblas/
COPY --from=build-graphblas /src/LAGraph/build/liblagraph.so.1 /lib/graphblas/
COPY --from=build-graphblas /src/LAGraph/build/liblagraph.so /lib/graphblas/

# get LAGraph benchmarks
COPY --from=build-graphblas /src/LAGraph/build/src/benchmark/mxm_serial_demo /benchmark/
COPY --from=build-graphblas /src/LAGraph/build/src/benchmark/tc_large_matrices_demo /benchmark/
COPY --from=build-graphblas /src/LAGraph/build/src/benchmark/bfs_demo /benchmark/

# get MLIR
COPY --from=build-comet /src/COMET/llvm/build/bin/mlir-cpu-runner /bin/llvm/
COPY --from=build-comet /src/COMET/llvm/build/bin/mlir-opt /bin/llvm/
COPY --from=build-comet /src/COMET/llvm/build/lib/libmlir_c_runner_utils.so /lib/llvm/
COPY --from=build-comet /src/COMET/llvm/build/lib/libmlir_c_runner_utils.so.17git /lib/llvm/
COPY --from=build-comet /src/COMET/llvm/build/lib/libmlir_runner_utils.so /lib/llvm/
COPY --from=build-comet /src/COMET/llvm/build/lib/libmlir_runner_utils.so.17git /lib/llvm/
COPY --from=build-comet /src/COMET/llvm/build/lib/libmlir_float16_utils.so /lib/llvm/
COPY --from=build-comet /src/COMET/llvm/build/lib/libmlir_float16_utils.so.17git /lib/llvm/

# get COMET
COPY --from=build-comet /src/COMET/build/bin/comet-opt /bin/comet/
COPY --from=build-comet /src/COMET/build/lib/libcomet_runner_utils.so /lib/comet/
COPY --from=build-comet /src/COMET/build/lib/libcomet_runner_utils.so.17git /lib/comet/

# get BLIS (COMET dependency)
COPY --from=build-comet /src/COMET/install/lib/libblis.so.4.0.0 /lib/comet/
COPY --from=build-comet /src/COMET/install/lib/libblis.so.4 /lib/comet/
COPY --from=build-comet /src/COMET/install/lib/libblis.so /lib/comet/

# get scripts for running expts.
RUN mkdir /AE && \ 
    mkdir /AE/benchmarks && \
    mkdir /AE/data && \
    mkdir /AE/results && \
    mkdir /AE/scripts && \
    mkdir /AE/src 
COPY --from=build-comet /src/COMET/AE/benchmarks/. /AE/benchmarks
COPY --from=build-comet /src/COMET/AE/scripts/. /AE/scripts
COPY --from=build-comet /src/COMET/AE/src/. /AE/src
COPY --from=build-comet /src/COMET/AE/README.md /AE

WORKDIR /AE




