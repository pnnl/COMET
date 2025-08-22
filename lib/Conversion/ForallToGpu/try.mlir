"func.func"() <{function_type = () -> (), sym_name = "main"}> ({
  %0 = "arith.constant"() <{value = 1.700000e+00 : f64}> : () -> f64
  %1 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
  %2 = "arith.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "arith.constant"() <{value = 10 : index}> : () -> index
  %4 = "arith.constant"() <{value = 9 : index}> : () -> index
  %5 = "arith.constant"() <{value = 8 : index}> : () -> index
  %6 = "arith.constant"() <{value = 7 : index}> : () -> index
  %7 = "arith.constant"() <{value = 6 : index}> : () -> index
  %8 = "arith.constant"() <{value = 5 : index}> : () -> index
  %9 = "arith.constant"() <{value = 4 : index}> : () -> index
  %10 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %11 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %12 = "arith.constant"() <{value = 3 : index}> : () -> index
  %13 = "arith.constant"() <{value = 2 : index}> : () -> index
  %14 = "arith.constant"() <{value = -1 : index}> : () -> index
  %15 = "arith.constant"() <{value = 1 : index}> : () -> index
  %16 = "arith.constant"() <{value = 0 : index}> : () -> index
  %17 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<13xindex>
  %18 = "memref.cast"(%17) : (memref<13xindex>) -> memref<*xindex>
  "func.call"(%11, %16, %14, %15, %14, %18, %10) <{callee = @read_input_sizes_2D_f64}> {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
  %19 = "memref.load"(%17, %16) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %20 = "memref.load"(%17, %15) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %21 = "memref.load"(%17, %13) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %22 = "memref.load"(%17, %12) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %23 = "memref.load"(%17, %9) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %24 = "memref.load"(%17, %8) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %25 = "memref.load"(%17, %7) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %26 = "memref.load"(%17, %6) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %27 = "memref.load"(%17, %5) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %28 = "memref.load"(%17, %4) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %29 = "memref.load"(%17, %3) <{nontemporal = false}> : (memref<13xindex>, index) -> index
  %30 = "memref.alloc"(%19) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %19, %15) ({
  ^bb0(%arg19: index):
    "memref.store"(%2, %30, %arg19) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %31 = "memref.cast"(%30) : (memref<?xi64>) -> memref<*xi64>
  %32 = "memref.alloc"(%20) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %20, %15) ({
  ^bb0(%arg18: index):
    "memref.store"(%2, %32, %arg18) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %33 = "memref.cast"(%32) : (memref<?xi64>) -> memref<*xi64>
  %34 = "memref.alloc"(%21) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %21, %15) ({
  ^bb0(%arg17: index):
    "memref.store"(%2, %34, %arg17) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %35 = "memref.cast"(%34) : (memref<?xi64>) -> memref<*xi64>
  %36 = "memref.alloc"(%22) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %22, %15) ({
  ^bb0(%arg16: index):
    "memref.store"(%2, %36, %arg16) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %37 = "memref.cast"(%36) : (memref<?xi64>) -> memref<*xi64>
  %38 = "memref.alloc"(%23) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %23, %15) ({
  ^bb0(%arg15: index):
    "memref.store"(%2, %38, %arg15) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %39 = "memref.cast"(%38) : (memref<?xi64>) -> memref<*xi64>
  %40 = "memref.alloc"(%24) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %24, %15) ({
  ^bb0(%arg14: index):
    "memref.store"(%2, %40, %arg14) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %41 = "memref.cast"(%40) : (memref<?xi64>) -> memref<*xi64>
  %42 = "memref.alloc"(%25) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %25, %15) ({
  ^bb0(%arg13: index):
    "memref.store"(%2, %42, %arg13) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %43 = "memref.cast"(%42) : (memref<?xi64>) -> memref<*xi64>
  %44 = "memref.alloc"(%26) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xi64>
  "scf.for"(%16, %26, %15) ({
  ^bb0(%arg12: index):
    "memref.store"(%2, %44, %arg12) <{nontemporal = false}> : (i64, memref<?xi64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %45 = "memref.cast"(%44) : (memref<?xi64>) -> memref<*xi64>
  %46 = "memref.alloc"(%27) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>
  "scf.for"(%16, %27, %15) ({
  ^bb0(%arg11: index):
    "memref.store"(%1, %46, %arg11) <{nontemporal = false}> : (f64, memref<?xf64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %47 = "memref.cast"(%46) : (memref<?xf64>) -> memref<*xf64>
  "func.call"(%11, %16, %14, %15, %14, %31, %33, %35, %37, %39, %41, %43, %45, %47, %10) <{callee = @read_input_2D_f64_i64}> {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xi64>, memref<*xi64>, memref<*xi64>, memref<*xi64>, memref<*xi64>, memref<*xi64>, memref<*xi64>, memref<*xi64>, memref<*xf64>, i32) -> ()
  %48 = "memref.alloc"(%29) <{alignment = 32 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?x4xf64>
  "scf.for"(%16, %29, %15) ({
  ^bb0(%arg9: index):
    "scf.for"(%16, %9, %15) ({
    ^bb0(%arg10: index):
      "memref.store"(%0, %48, %arg9, %arg10) <{nontemporal = false}> : (f64, memref<?x4xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %49 = "memref.alloc"(%28) <{alignment = 32 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?x4xf64>
  "scf.for"(%16, %28, %15) ({
  ^bb0(%arg7: index):
    "scf.for"(%16, %9, %15) ({
    ^bb0(%arg8: index):
      "memref.store"(%1, %49, %arg7, %arg8) <{nontemporal = false}> : (f64, memref<?x4xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %50 = "arith.constant"() <{value = 0 : index}> : () -> index
  %51 = "arith.constant"() <{value = 1 : index}> : () -> index
  %52 = "arith.constant"() <{value = 0 : index}> : () -> index
  %53 = "arith.constant"() <{value = 4 : index}> : () -> index
  %54 = "arith.constant"() <{value = 1 : index}> : () -> index
  %55 = "arith.constant"() <{value = 0 : index}> : () -> index
  %56 = "arith.constant"() <{value = 1 : index}> : () -> index
  %57 = "arith.constant"() <{value = 32 : index}> : () -> index
  %58 = "arith.muli"(%51, %56) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  %59 = "arith.muli"(%54, %57) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  "scf.parallel"(%50, %52, %28, %53, %58, %59) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%arg0: index, %arg1: index):
    %61 = "affine.min"(%28, %arg0) <{map = affine_map<(d0, d1) -> (1, d0 - d1)>}> : (index, index) -> index
    %62 = "affine.min"(%53, %arg1) <{map = affine_map<(d0, d1) -> (32, d0 - d1)>}> : (index, index) -> index
    "scf.for"(%55, %62, %54) ({
    ^bb0(%arg2: index):
      %63 = "arith.addi"(%arg2, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %64 = "arith.index_cast"(%65) : (i64) -> index
      %65 = "memref.load"(%38, %68) <{nontemporal = false}> : (memref<?xi64>, index) -> i64
      %66 = "arith.index_cast"(%67) : (i64) -> index
      %67 = "memref.load"(%38, %arg0) <{nontemporal = false}> : (memref<?xi64>, index) -> i64
      %68 = "arith.addi"(%arg0, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %69 = "memref.load"(%49, %arg0, %63) <{nontemporal = false}> : (memref<?x4xf64>, index, index) -> f64
      %70 = "arith.constant"() <{value = 0 : index}> : () -> index
      %71 = "arith.constant"() <{value = 32 : index}> : () -> index
      %72 = "arith.muli"(%15, %71) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %73 = "scf.for"(%66, %64, %72, %69) ({
      ^bb0(%arg3: index, %arg4: f64):
        %74 = "affine.min"(%64, %arg3) <{map = affine_map<(d0, d1) -> (32, d0 - d1)>}> : (index, index) -> index
        %75 = "scf.for"(%70, %74, %15, %arg4) ({
        ^bb0(%arg5: index, %arg6: f64):
          %76 = "arith.addi"(%arg5, %arg3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %77 = "memref.load"(%40, %76) <{nontemporal = false}> : (memref<?xi64>, index) -> i64
          %78 = "arith.index_cast"(%77) : (i64) -> index
          %79 = "memref.load"(%46, %76) <{nontemporal = false}> : (memref<?xf64>, index) -> f64
          %80 = "memref.load"(%48, %78, %63) <{nontemporal = false}> : (memref<?x4xf64>, index, index) -> f64
          %81 = "arith.mulf"(%79, %80) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          %82 = "arith.addf"(%arg6, %81) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          "scf.yield"(%82) : (f64) -> ()
        }) {blockSize = 32 : ui32} : (index, index, index, f64) -> f64
        "scf.yield"(%75) : (f64) -> ()
      }) {reduceDim} : (index, index, index, f64) -> f64
      "memref.store"(%73, %49, %arg0, %63) <{nontemporal = false}> : (f64, memref<?x4xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) {blockSize = 32 : ui32} : (index, index, index) -> ()
    "scf.reduce"() : () -> ()
  }) {mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>], parallelDim = ["dimY_grid", "dimX_grid"]} : (index, index, index, index, index, index) -> ()
  %60 = "memref.cast"(%49) : (memref<?x4xf64>) -> memref<*xf64>
  "func.call"(%60) <{callee = @comet_print_memref_f64}> : (memref<*xf64>) -> ()
  "func.return"() : () -> ()
}) : () -> ()