source scripts/paths.sh

py_script="scripts/py2.triangle_counting.py"
GraphX_TC_Burkhardt_code="src/triangle_counting.Burkhardt.ta"
GraphX_TC_Cohen_code="src/triangle_counting.Cohen.ta"
GraphX_TC_Sandia_LL_code="src/triangle_counting.Sandia_LL.ta"
GraphX_TC_Sandia_UU_code="src/triangle_counting.Sandia_UU.ta"
csv_file="results/triangle_counting.csv"
LAGraph_exe="tc_large_matrices_demo"
plot_script="scripts/plot2.triangle_counting.speedup.py"


# Run benchmark
python ${py_script} \
       ${GraphX_TC_Burkhardt_code} \
       ${GraphX_TC_Cohen_code} \
       ${GraphX_TC_Sandia_LL_code} \
       ${GraphX_TC_Sandia_UU_code} \
       ${csv_file} \
       ${LAGraph_exe}

# Generate the figure under results/
python ${plot_script} ${csv_file}
