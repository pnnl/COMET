if [ $# -eq 1 ] && [ "$1" = "test" ]; then
  source scripts/paths.test_version.sh
else
  source scripts/paths.sh
fi

py_script="scripts/py3.bfs.py"
GraphX_code="src/bfs.mlir"
csv_file="results/bfs.csv"
LAGraph_exe="bfs_demo"
plot_script="scripts/plot3.bfs.speedup.py"


# Run benchmark
python3 ${py_script} ${GraphX_code} ${csv_file} ${LAGraph_exe}

# Generate the figure under results/
python3 ${plot_script} ${csv_file}
