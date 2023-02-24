module  {
func @elwise_coo_dense_coo() {
%c00 = constant 0: index
%c01 = constant 1: index
%0 = "ta.dynamic_index_label"(%c00, %c01) : (index, index) -> !ta.range

%c10 = constant 0: index
%c11 = constant 1: index
%1 = "ta.dynamic_index_label"(%c10, %c11) : (index, index) -> !ta.range

%2 = "ta.sparse_tensor_decl"( %0, %1) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
"ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME2", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()

%3 = "ta.dense_tensor_decl"( %0, %1) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
"ta.fill"(%3) {value = 2.7 : f64 } : (tensor<?x?xf64>) -> ()

%4 = "ta.sparse_tensor_decl"( %0, %1) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>

%5 = "ta.elews_mul"(%2,%3,  %0, %1) {__alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["COO", "Dense", "COO"], indexing_maps = [affine_map<( d0, d1) -> ( d0, d1)>, affine_map<( d0, d1) -> ( d0, d1)>, affine_map<( d0, d1) -> ( d0, d1)>], semiring = "noop_times"} : (tensor<?x?xf64>, tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>

"ta.set_op"(%5, %4) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()

"ta.print"(%4) : (tensor<?x?xf64>) -> ()

"ta.return"(): () -> ()
}
}

