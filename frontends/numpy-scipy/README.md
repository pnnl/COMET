
# cometpy

Comet Domain Specific Compiler for NumPy and Scipy frontends.

Latest Version : v0.2

Requirements:
1. COMET backend installed on your computer
2. python 3.6 and above
3. numpy
4. scipy >=1.9
5. jinja2


Tested OS support - Linux and macOS

# "COMET" / Domain specific COMpiler for Extreme Targets

The COMET compiler consists of a Domain Specific Language (DSL) for sparse and dense tensor algebra computations, a progressive lowering process to map high-level operations to low-level architectural resources, a series of optimizations performed in the lowering process, and various IR dialects to represent key concepts, operations, and types at each level of the multi-level IR. At each level of the IR stack, COMET performs different optimizations and code transformations. Domain-specific, hardware- agnostic optimizations that rely on high-level semantic information are applied at high-level IRs. These include reformulation of high-level operations in a form that is amenable for execution on heterogeneous devices (e.g., rewriting Tensor contraction operations as Transpose-Transpose-GEMM-Transpose) and automatic parallelization of high-level primitives (e.g., tiling for thread- and task-level parallelism).

More information about the COMET compiler can be found at: 
1. COMET's source code - https://github.com/pnnl/COMET
2. COMET’s documentation - https://pnnl-comet.readthedocs.io/en/latest/


# Installation and Testing:

1. Installing COMET's python package:
    * Install [COMET backend](../../README.md)
    * Set environmental variables for paths to COMET and LLVM:
    ```
    COMETPY_COMET_PATH=/path/to/comet/build/
    COMETPY_LLVM_PATH=/path/to/llvm/build/
    ```
    * In directory `frontends/numpy-scipy` run:
    ```
    python3 -m pip install .
    ```
    * This will also install the package dependencies if not already installed

2. Run the integration tests to make sure the installation was successfull
    ```
    cd integration_tests
    python3 numpy_integration.py
    ```

3. Using COMET compiler as a backend to lower and execute test methods:
    -   Import the Comet Python package in your python script using - "import cometpy as comet"
    -   To compile a test method using COMET, add the decorator "@comet.compile(flags=...)"
        before the method.
    -   The "numpy" keyword in a target method needs to be replace with "comet"
        e.g. "comet.einsum" , "comet.multiply"

    -   For the actual "einsum" computation, follow the convention - "comet.einsum()"
    

4. Currently the code included in a comet kernel (i.e. functions annotated with @comet.compile) should take as input only ndarrays and only operate on them with the following supported operations: 
    - +, -, *, @ which work as intended for ndarrays
    - A.transpose()  or comet.einsum('ij->ji', A) for transpose operations
    - comet.einsum() for tensor contraction operations

5. Control flow operations are not supported and should be handled outside of the kernel call