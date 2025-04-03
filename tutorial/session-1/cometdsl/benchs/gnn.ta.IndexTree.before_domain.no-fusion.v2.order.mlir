/// GNN kernel A = B * C * D
/// B is sparse
/// T[i, h] = B[i, k] * C[k, h];
/// A[i, j] = T[i, h] * D[h, j];
/// 
/// void no_fution_index_tree()
/// {
///     for (h = 0 to NH) {
///         for (i = 0 to NI) {
///             for (k = 0 to NK) {
///                 T[i, h] += B[i, k] * C[k, h];
///             }
///         }
///     }
///     for (h = 0 to NH) {
///         for (i = 0 to NI) {
///             for (j = 0 to NJ) {
///                 A[i, j] += T[i, h] * D[h, j];
///             }
///         }
///     }
/// }
/// void partial_fusion_index_tree()
/// {
///     for (h = 0 to NH) {
///         for (i = 0 to NI) {
///             for (k = 0 to NK) {
///                 t += B[i, k] * C[k, h];
///             }
///             for (j = 0 to NJ) {
///                 A[i, j] += t * D[h, j];
///             }
///             t = 0;
///         }
///     }
/// }

func.func @main() {
  %0 = "ta.index_label"() : () -> !ta.index
  %1 = "ta.index_label"() : () -> !ta.index
  %2 = "ta.index_label"() : () -> !ta.index
  %3 = "ta.index_label"() : () -> !ta.index
  %4 = "ta.spTensor_decl"() <{format = "CSR", temporal_tensor = false}> : () -> !ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>  /// %4 = B
  %c0 = arith.constant 0 : index
  %5 = "ta.dim"(%4, %c0) : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, index) -> index
  %c1 = arith.constant 1 : index
  %6 = "ta.dim"(%4, %c1) : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, index) -> index
  %7 = "ta.dense_tensor_decl"(%6) <{format = "Dense"}> : (index) -> tensor<?x16xf64>  /// %7 = C
  %8 = "ta.dense_tensor_decl"() <{format = "Dense"}> : () -> tensor<16x16xf64>  /// %8 = D
  %9 = "ta.dense_tensor_decl"(%5) <{format = "Dense"}> : (index) -> tensor<?x16xf64>  /// %9 = A
  %10 = "ta.dense_tensor_decl"(%5) <{format = "Dense"}> : (index) -> tensor<?x16xf64>  /// %10 = T
  "ta.fill_from_file"(%4) <{filename = "SPARSE_FILE_NAME0", readMode = 1 : i32}> : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>) -> ()
  "ta.fill"(%7) <{value = 1.200000e+00 : f64}> : (tensor<?x16xf64>) -> ()
  "ta.fill"(%8) <{value = 3.400000e+00 : f64}> : (tensor<16x16xf64>) -> ()
  "ta.fill"(%9) <{value = 0.000000e+00 : f64}> : (tensor<?x16xf64>) -> ()
  "ta.fill"(%10) <{value = 0.000000e+00 : f64}> : (tensor<?x16xf64>) -> ()
  %11 = "ta.getTime"() : () -> f64
  %12 = "it.itree"(%10) <{operandSegmentSizes = array<i32: 1, 0>}> ({
  ^bb0(%arg0: tensor<?x16xf64>):
    %16 = "it.RootOp"() : () -> !it.index_tree
    %i = "it.IndexOp"(%16) : (!it.index_tree) -> !it.index  /// %17 = h
    %k = "it.IndexOp"(%i) : (!it.index) -> !it.index  /// %18 = i
    %h = "it.IndexOp"(%k) : (!it.index) -> !it.index  /// %19 = k
    %crd, %pos = "it.IndexToTensorDim"(%arg0, %i) <{dim = 0 : ui32}> : (tensor<?x16xf64>, !it.index) -> (index, index)
    %crd_0, %pos_1 = "it.IndexToTensorDim"(%arg0, %h, %pos) <{dim = 1 : ui32}> : (tensor<?x16xf64>, !it.index, index) -> (index, index)
    %20 = "it.LHSOperandOp"(%arg0, %pos, %pos_1, %crd, %crd_0) : (tensor<?x16xf64>, index, index, index, index) -> !it.operand
    %crd_2, %pos_3 = "it.IndexToTensorDim"(%4, %i) <{dim = 0 : ui32}> : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, !it.index) -> (index, index)
    %crd_4, %pos_5 = "it.IndexToTensorDim"(%4, %k, %pos_3) <{dim = 1 : ui32}> : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, !it.index, index) -> (index, index)
    %21 = "it.OperandOp"(%4, %pos_3, %pos_5, %crd_2, %crd_4) : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, index, index, index, index) -> !it.operand
    %crd_6, %pos_7 = "it.IndexToTensorDim"(%7, %k) <{dim = 0 : ui32}> : (tensor<?x16xf64>, !it.index) -> (index, index)
    %crd_8, %pos_9 = "it.IndexToTensorDim"(%7, %h, %pos_7) <{dim = 1 : ui32}> : (tensor<?x16xf64>, !it.index, index) -> (index, index)
    %22 = "it.OperandOp"(%7, %pos_7, %pos_9, %crd_6, %crd_8) : (tensor<?x16xf64>, index, index, index, index) -> !it.operand
    %23 = "it.ComputeOp"(%h, %20, %21, %22) <{compute_missing = false, operandSegmentSizes = array<i32: 1, 1, 2, 0>, semiring = "plusxy_times"}> : (!it.index, !it.operand, !it.operand, !it.operand) -> tensor<?x16xf64>
    it.yield %23 : tensor<?x16xf64>
  }) : (tensor<?x16xf64>) -> tensor<?x16xf64>
  "ta.set_op"(%12, %10) {__beta__ = 0.000000e+00 : f64} : (tensor<?x16xf64>, tensor<?x16xf64>) -> ()
  %13 = "it.itree"(%9) <{operandSegmentSizes = array<i32: 1, 0>}> ({
  ^bb0(%arg0: tensor<?x16xf64>):
    %16 = "it.RootOp"() : () -> !it.index_tree
    %i = "it.IndexOp"(%16) : (!it.index_tree) -> !it.index  /// %17 = h
    %h = "it.IndexOp"(%i) : (!it.index) -> !it.index  /// %18 = i
    %j = "it.IndexOp"(%h) : (!it.index) -> !it.index  /// %19 = j
    %crd, %pos = "it.IndexToTensorDim"(%arg0, %i) <{dim = 0 : ui32}> : (tensor<?x16xf64>, !it.index) -> (index, index)
    %crd_0, %pos_1 = "it.IndexToTensorDim"(%arg0, %j, %pos) <{dim = 1 : ui32}> : (tensor<?x16xf64>, !it.index, index) -> (index, index)
    %20 = "it.LHSOperandOp"(%arg0, %pos, %pos_1, %crd, %crd_0) : (tensor<?x16xf64>, index, index, index, index) -> !it.operand
    %crd_2, %pos_3 = "it.IndexToTensorDim"(%10, %i) <{dim = 0 : ui32}> : (tensor<?x16xf64>, !it.index) -> (index, index)
    %crd_4, %pos_5 = "it.IndexToTensorDim"(%10, %h, %pos_3) <{dim = 1 : ui32}> : (tensor<?x16xf64>, !it.index, index) -> (index, index)
    %21 = "it.OperandOp"(%10, %pos_3, %pos_5, %crd_2, %crd_4) : (tensor<?x16xf64>, index, index, index, index) -> !it.operand
    %crd_6, %pos_7 = "it.IndexToTensorDim"(%8, %h) <{dim = 0 : ui32}> : (tensor<16x16xf64>, !it.index) -> (index, index)
    %crd_8, %pos_9 = "it.IndexToTensorDim"(%8, %j, %pos_7) <{dim = 1 : ui32}> : (tensor<16x16xf64>, !it.index, index) -> (index, index)
    %22 = "it.OperandOp"(%8, %pos_7, %pos_9, %crd_6, %crd_8) : (tensor<16x16xf64>, index, index, index, index) -> !it.operand
    %23 = "it.ComputeOp"(%j, %20, %21, %22) <{compute_missing = false, operandSegmentSizes = array<i32: 1, 1, 2, 0>, semiring = "plusxy_times"}> : (!it.index, !it.operand, !it.operand, !it.operand) -> tensor<?x16xf64>
    it.yield %23 : tensor<?x16xf64>
  }) : (tensor<?x16xf64>) -> tensor<?x16xf64>
  "ta.set_op"(%13, %9) {__beta__ = 0.000000e+00 : f64} : (tensor<?x16xf64>, tensor<?x16xf64>) -> ()
  %14 = "ta.getTime"() : () -> f64
  %15 = "ta.reduce"(%9) : (tensor<?x16xf64>) -> f64
  "ta.print"(%15) : (f64) -> ()
  "ta.print_elapsed_time"(%11, %14) : (f64, f64) -> ()
  return
}