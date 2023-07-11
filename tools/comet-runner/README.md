# comet-runner

## Compilation Flow
 ![](comet-dsl.svg)

## Building Required Tools

### llvm-dis-7, llvm-spirv-v7.0.0-1

    git clone --branch release/7.x --depth 1 https://github.com/llvm/llvm-project.git llvm-7
    # depth 1 optional, but recommended for much faster clone
    cd llvm-7/llvm/projects
    git clone --branch llvm_release_70 https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
    mkdir ../../build
    cd ../../build
    cmake -GNinja ../llvm \
        -DLLVM_ENABLE_PROJECTS=clang \
        -DLLVM_TARGETS_TO_BUILD=host
    ninja llvm-spirv
    ninja llvm-dis

### llvm-as-hls, llvm-link-hls

    git clone https://github.com/Xilinx/HLS.git llvm-hls
    cd llvm-hls
    git submodule update --init
    ./build-hls-llvm-project.sh
