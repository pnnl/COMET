#/bin/sh

export COMET_SRC=/Users/kest268/projects/COMET/COMET/
OPENBLAS="/opt/homebrew/opt/openblas/"

export LDFLAGS="-L$OPENBLAS/lib"
export CPPFLAGS="-I$OPENBLAS/include"
export PKG_CONFIG_PATH="$OPENBLAS/lib/pkgconfig"

export PYTHON_EXECUTABLE=$(which python3.12)
export COMETPY_COMET_PATH=$COMET_SRC/build/ 
export COMETPY_LLVM_PATH=$COMET_SRC/llvm/build

export COMET_BIN_DIR=$COMETPY_COMET_PATH/bin       #${COMET_SRC}/build/bin
export COMET_LIB_DIR=$COMETPY_COMET_PATH/lib       #${COMET_SRC}/build/lib
export MLIR_BIN_DIR=$COMETPY_LLVM_PATH/bin   #${COMET_SRC}/llvm/bin
export MLIR_LIB_DIR=$COMETPY_LLVM_PATH/lib   #${COMET_SRC}/llvm/lib

export OMP_NUM_THREADS=1