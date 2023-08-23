if [ $# -eq 1 ] && [ "$1" = "test" ]; then
  source scripts/paths.test_version.sh
else
  source scripts/paths.sh
fi

py_script="scripts/py1.masked_spgemm.py"
GraphX_ta="src/masked_SpGEMM.ta"
GraphX_scf="src/masked_SpGEMM.mlir"
csv_file="results/masked_spgemm.csv"
LAGraph_exe="mxm_serial_demo"
plot_script="scripts/plot1.masked_spgemm.speedup.py"


# Run benchmark
python3 ${py_script} ${GraphX_ta} ${GraphX_scf} ${csv_file} ${LAGraph_exe}

# Generate the figure under results/
python3 ${plot_script} ${csv_file}
