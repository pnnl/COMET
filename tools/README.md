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

## Build MCL runtime
Follow directions in `/runtimes/mcl` directory

## Running FPGA kernel generated from COMET
1. Generate the runnable from COMET
```bash
comet-op --target=FPGA --convert-ta-to-it --convert-to-loops --convert-to-llvm <path/to/input.ta> --xclbin_path=<path/to/xclbin> &> run.mlir 
```
This command will generate a `.bin` file at `<path/to/input.ta__spv_<kernel_name>.bin>`
and print the host code to `run.mlir`.

Note that the path passed to ``--xclbin-path`` needs to be the path to the final xclbin file generated after all the following steps have been run.


2. Converting the `.bin` file to `.xclbin` using the `spirv-to-xclbin.py` script
```
python3 spirv-to-xclbin.py <path/to/input.ta__spv_<kernel_name>>.bin -k <kernel_name> -l <platform_id> -o <path/to/xclbin>
``` 
This should generate (among others) the `.xclbin` file at the location <path/to/xclbin>.xlcbin

3. Running on the FPGA

    a. First, start MCL

    b. Run the host code

```bash
<mcl/bin/path>/mcl_scheduler &
```

```
<path/to/llvm/bin>/mlir-cpu-runner run.mlir -O3 -e main -entry-point-result=void -shared-libs=<comet/utility/library/dir>/libcomet_runner_utils.so,<mlir/utility/library/dir>/libomp.so
```
