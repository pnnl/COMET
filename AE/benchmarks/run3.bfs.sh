source scripts/paths.sh

py_script="scripts/py3.bfs.py"
GraphX_code="src/bfs.mlir"
csv_file="results/bfs.csv"
LAGraph_exe="bfs_demo"
plot_script="scripts/plot3.bfs.speedup.py"


# Run benchmark
python ${py_script} ${GraphX_code} ${csv_file} ${LAGraph_exe}

# Generate the figure under results/
python ${plot_script} ${csv_file}
