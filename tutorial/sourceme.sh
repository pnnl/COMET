#/bin/sh

export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"

export PYTHON_EXECUTABLE=$(which python3.x)
export COMET_SRC=`pwd`/../
export COMETPY_COMET_PATH=`pwd`/../build/
export COMETPY_LLVM_PATH=`pwd`/../llvm/build/

export COMET_BIN_DIR=$COMETPY_COMET_PATH/bin       #${COMET_SRC}/build/bin
export COMET_LIB_DIR=$COMETPY_COMET_PATH/lib       #${COMET_SRC}/build/lib
export MLIR_BIN_DIR=$COMETPY_LLVM_PATH/bin   #${COMET_SRC}/llvm/bin
export MLIR_LIB_DIR=$COMETPY_LLVM_PATH/lib   #${COMET_SRC}/llvm/lib