if [ $# -eq 1 ] && [ "$1" = "test" ]; then
  source scripts/paths.test_version.sh
else
  source scripts/paths.sh
fi

py_script="scripts/py2.triangle_counting.py"
GraphX_TC_Burkhardt_ta="src/triangle_counting.Burkhardt.ta"
GraphX_TC_Burkhardt_scf="src/triangle_counting.Burkhardt.mlir"
GraphX_TC_Cohen_ta="src/triangle_counting.Cohen.ta"
GraphX_TC_Cohen_scf="src/triangle_counting.Cohen.mlir"
GraphX_TC_Sandia_LL_ta="src/triangle_counting.Sandia_LL.ta"
GraphX_TC_Sandia_LL_scf="src/triangle_counting.Sandia_LL.mlir"
GraphX_TC_Sandia_UU_ta="src/triangle_counting.Sandia_UU.ta"
GraphX_TC_Sandia_UU_scf="src/triangle_counting.Sandia_UU.mlir"
csv_file="results/triangle_counting.csv"
LAGraph_exe="tc_large_matrices_demo"
plot_script="scripts/plot2.triangle_counting.speedup.py"


# Run benchmark
python3 ${py_script} \
       ${GraphX_TC_Burkhardt_ta} \
       ${GraphX_TC_Burkhardt_scf} \
       ${GraphX_TC_Cohen_ta} \
       ${GraphX_TC_Cohen_scf} \
       ${GraphX_TC_Sandia_LL_ta} \
       ${GraphX_TC_Sandia_LL_scf} \
       ${GraphX_TC_Sandia_UU_ta} \
       ${GraphX_TC_Sandia_UU_scf} \
       ${csv_file} \
       ${LAGraph_exe}

# Generate the figure under results/
python3 ${plot_script} ${csv_file}
