
Requirements:
1. python 3 and above
2. Comet python package 
3. Ubuntu based Linux distro or macOS >= Catalina

Python packages required:
1. Numpy
2. ast
3. inspect
4. jinja2

Installation and Testing:
1. Getting COMET:
   - Install the comet package using pip (or) pip3 python package manager as:
    (a) For macOS users:
        python3 -m pip install cometpy
    (b) For Linux users:
        python3 -m pip install cometpy-lnx

2. To run all the unit tests in the numpy/test directory, run the script numpy_integration:
      python3 numpy_integration.py 
    - To run an individual test case use:
      python3 <file_name>.py

3.  The test cases in "test" directory follow the naming convention : "test_*.py"
    - One can add a test case using the above convention

4. Using COMET to lower test methods:
    -   Import comet in your python script as shown in the test cases
        python code : "from comet_pkg import comet"
    -   To compile a test method using comet, add the decorator "@comet.compile(flags=...)"
        before the method as shown in the test cases.
    -   For the actual "einsum" computation, follow the convention - "comet.einsum()" in the code
        as shown in the test cases.
        (i) "numpy" computations in a target method need to be replaced with keyword "comet"- 
            e.g. "comet.einsum"
        (ii) Other supported numpy operations - "numpy.multiply"

For Developers:

1. For modifying and or testing the python-to-TA dialect mapping source code:
   - The mapping source code is in "comet.py" in the parent directory.
     (i) The supporting scripts for "comet.py" are in MLIRGen directory.
         Required scripts - 
         (a) lowering.py
              This script lowers and executes the generated TA dialect
              (Makes calls to "comet-opt", "mlir-opt", "mlir-translate", "lli)
         (b) PyMLIRGen.py
              This script generates the TA dialect operations in the form of
              strings.
         (c) types_mlir.py
             Utility script for generating some Tensor Algebra (TA) dialect operations
         (d) utils.py
             Utility script containing data structures to store information
         
         *Other scripts in MLIRGen (lowering_with_cinter.py and llvmlite_jit.py) and the 
          "bridge" and "core" subdirectories, are meant for interfacing python with C++ 
           and lowering through functions. Currently they are not being used.       
   
2. The code for lowering and execution is in "lowering.py" in MLIRGen directory.
    This script depends on external executables as mentioned above.
    
