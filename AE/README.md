# Artifact Evaluation
## Overall
All command is running under `AE/` (run `$ cd AE/` first). The `AE/scripts/paths.sh` sets all basic paths that are relative to `AE/`, NOT `AE/scripts/`. If COMET is not built under default paths, please feel free to change those paths accordingly.
## Software Environment
Please run
```shell
$ bash scripts/sh0.install_libraries.sh
```
to install Python libraries.

Please build LAGraph under `AE/LAGraph/build/`, which requires the [GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) library. In general, LAGraph can be built by run
```shell
$ cd AE/LAGraph/build/
$ GRAPHBLAS_ROOT=/directory/contains/libgraphblas.so cmake ..
$ make 
```

## Datasets
```shell
$ bash scripts/sh1.get_matrices.sh
```
It downloads all dataset under `AE/data/`.

## Quick Start
After datasets are ready, run
```shell
$ bash benchmarks/run0.quick_run.sh test
```
It runs the Masked SpGEMM benchmarks of GraphX and LAGraph using 8 small matrices, and generates a png figure under `AE/results/`.
Depending on the machines, this quick start takes about 10 minutes.

## Main Results
For full results of Masked SpGEMM with 10 matrices (8 small ones plus 2 large ones), run
```shell
$ bash benchmarks/run1.masked_spgemm.sh test
```

For results of Triangle Counting, run
```shell
$ bash benchmarks/run2.triangle_counting.sh test
```

For results of Breadth-First Search (BFS), run
```shell
$ bash benchmarks/run3.bfs.sh test
```
Each benchmark will generates a png figure under `AE/results/`.