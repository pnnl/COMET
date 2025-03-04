# COMET Compilation Tools


## SPIRV-LLVM Translator
1. If not already done, initialize the relevant submodules
```bash
git submodule update --init .
```
2. Create a link from spirv-llvm-translate to llvm-spirv projects directory
```bash
ln -s ${PWD}/spirv-llvm-translate llvm-spirv/llvm/projects
```
3. Build LLVM-SPIRV translator
```bash
mkdir llvm-spirv/build
cd llvm-spirv/build
cmake -GNinja ../llvm \
    -DLLVM_ENABLE_PROJECTS=clang \
    -DLLVM_TARGETS_TO_BUILD=host
ninja llvm-spirv
ninja llvm-dis
```