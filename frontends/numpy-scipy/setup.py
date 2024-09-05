import setuptools
import os
import subprocess

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("cometpy/cfg.py", "w") as cfg:
    if "COMETPY_COMET_PATH" in os.environ:
        if  os.path.exists(os.environ["COMETPY_COMET_PATH"]+"/bin/comet-opt"):
            cfg.write("comet_path = '{}'\n".format(os.environ["COMETPY_COMET_PATH"]))
        else:
            raise Exception("ERROR! {} does not exist".format(os.environ["COMETPY_COMET_PATH"]+"/bin/comet-opt"))

    else:
        print("COMETPY_COMET_PATH not specified attempting default path")
        if os.path.exists(os.path.abspath("../../build/")):
            cfg.write("comet_path = '{}'\n".format(os.path.abspath("../../build/")))
        else:
            raise Exception("ERROR! Path to COMET not found.")

    if "COMETPY_LLVM_PATH" in os.environ:
        if  os.path.exists(os.environ["COMETPY_LLVM_PATH"]+"/bin/clang"):
            cfg.write("llvm_path = '{}'\n".format(os.environ["COMETPY_LLVM_PATH"]))
        else:
            raise Exception("ERROR! {} does not exist".format(os.environ["COMETPY_LLVM_PATH"]+"/bin/clang"))

    else:
        print("COMETPY_LLVM_PATH not specified attempting default path")
        if os.path.exists(os.path.abspath("../../llvm/build/")):
            cfg.write("llvm_path = '{}'\n".format(os.path.abspath("../../llvm/build/")))
        else:
            raise Exception("ERROR! Path to COMET LLVM not found.")
    comet_opt = os.environ["COMETPY_COMET_PATH"]+"/bin/comet-opt"
    p = subprocess.run([comet_opt, "--target=GPU", "a"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if "Cannot find option named 'GPU'!" in p.stdout.decode():
        cfg.write("gpu_target_enabled = False")
    else:
        cfg.write("gpu_target_enabled = True")

setuptools.setup(
    name="cometpy",
    version="0.2",
    author="Polykarpos Thomadakis",
    author_email="polykarpos.thomadakis@pnnl.gov",
    description="Comet Domain Specific Compiler as Python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=["cometpy", "cometpy.MLIRGen"],
    package_dir={"cometpy": "cometpy"},
    install_requires=[
        'jinja2',
        'numpy',
        'scipy>=1.10'
    ],
    python_requires=">=3.8",
)