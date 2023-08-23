if [ $# -eq 1 ] && [ "$1" = "test" ]; then
  source scripts/paths.test_version.sh
else
  source scripts/paths.sh
fi



py_script="scripts/py0.quick_run.py"
GraphX_code="src/masked_SpGEMM.ta"
csv_file="results/quick_run.masked_spgemm.csv"
LAGraph_exe="mxm_serial_demo"
plot_script="scripts/plot0.quick_run.speedup.py"


# Run benchmark
python3 ${py_script} ${GraphX_code} ${csv_file} ${LAGraph_exe}

# Generate the figure under results/
python3 ${plot_script} ${csv_file}
