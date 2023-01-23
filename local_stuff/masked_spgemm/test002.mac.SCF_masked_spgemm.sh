
project_root="/people/peng599/pppp/clion/comet_ReACT-fusion-IT_mac"
matrices_root="/people/peng599/pppp/clion/react-eval_mac/matrices"

#input_code=( "mult_spgemm_CSRxCSR_oCSR.mask.mlir" "mult_spgemm_CSRxCSR_oCSR.mask.ta.manual.loops.v6.mlir")
matrices=("bcsstk29" "bcsstk17" "pdb1HYS" "cant" "rma10" "consph"
          "shipsec1" "pwtk" "cop20k_A" "mac_econ_fwd500" "scircuit")
rounds=10

### No Mask
echo ""
echo "#### No Mask ####"
code="mult_spgemm_CSRxCSR_oCSR.mask.mlir"
for matrix in "${matrices[@]}"; do
  python3 py000.runtime_masking.py "${project_root}" "${code}" "${matrices_root}/${matrix}/${matrix}.mtx" ${rounds}
done

### Mask
echo ""
echo "#### Mask ####"
code="mult_spgemm_CSRxCSR_oCSR.mask.ta.manual.loops.v6.mlir"
for matrix in "${matrices[@]}"; do
  python3 py000.runtime_masking.py "${project_root}" "${code}" "${matrices_root}/${matrix}/${matrix}.mtx" ${rounds}
done