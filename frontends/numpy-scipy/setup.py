import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("cometpy/cfg.py", "w+") as cfg:
    if "COMETPY_COMET_PATH" in os.environ:
        cfg.write("comet_path = '{}'\n".format(os.environ["COMETPY_COMET_PATH"]))
    else:
        print("COMETPY_COMET_PATH not specified attempting default path")
        if os.path.exists(os.path.abspath("../../build/")):
            cfg.write("comet_path = '{}'\n".format(os.path.abspath("../../build/")))
        else:
            print("ERROR! Path to COMET not found.")
            exit(0)

    if "COMETPY_LLVM_PATH" in os.environ:
        cfg.write("llvm_path = '{}'\n".format(os.environ["COMETPY_LLVM_PATH"]))
    else:
        print("COMETPY_LLVM_PATH not specified attempting default path")
        if os.path.exists(os.path.abspath("../../llvm/build/")):
            cfg.write("llvm_path = '{}'\n".format(os.path.abspath("../../llvm/build/")))
        else:
            print("ERROR! Path to COMET LLVM not found.")
            exit(0)


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
        'scipy>=1.9'
    ],
    python_requires=">=3.6",
)