source scripts/paths.sh

py_script="scripts/py1.masked_spgemm.py"
GraphX_code="src/masked_SpGEMM.ta"
csv_file="results/masked_spgemm.csv"
LAGraph_exe="mxm_serial_demo"
plot_script="scripts/plot1.masked_spgemm.speedup.py"


# Run benchmark
python ${py_script} ${GraphX_code} ${csv_file} ${LAGraph_exe}

# Generate the figure under results/
python ${plot_script} ${csv_file}
