module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(2 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(3 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(-1 : index) : i64
    %8 = llvm.mlir.constant(7 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.null : !llvm.ptr<i64>
    %11 = llvm.getelementptr %10[7] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %12 = llvm.ptrtoint %11 : !llvm.ptr<i64> to i64
    %13 = llvm.call @malloc(%12) : (i64) -> !llvm.ptr<i8>
    %14 = llvm.bitcast %13 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %15 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %9, %20[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.alloca %22 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %21, %23 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %24 = llvm.bitcast %23 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<(i64, ptr<i8>)> 
    %28 = llvm.insertvalue %24, %27[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_2D_f64(%4, %6, %5, %6, %7, %25, %24, %4) {filename = "SPARSE_FILE_NAME1"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %29 = llvm.getelementptr %14[%6] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %30 = llvm.load %29 : !llvm.ptr<i64>
    %31 = llvm.getelementptr %14[%3] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %32 = llvm.load %31 : !llvm.ptr<i64>
    %33 = llvm.getelementptr %14[%2] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %34 = llvm.load %33 : !llvm.ptr<i64>
    %35 = llvm.getelementptr %14[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %36 = llvm.load %35 : !llvm.ptr<i64>
    %37 = llvm.getelementptr %14[%1] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %38 = llvm.load %37 : !llvm.ptr<i64>
    %39 = llvm.mlir.constant(1 : index) : i64
    %40 = llvm.mlir.null : !llvm.ptr<i64>
    %41 = llvm.getelementptr %40[%30] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %42 = llvm.ptrtoint %41 : !llvm.ptr<i64> to i64
    %43 = llvm.call @malloc(%42) : (i64) -> !llvm.ptr<i8>
    %44 = llvm.bitcast %43 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %45 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.insertvalue %44, %45[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %44, %46[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %50 = llvm.insertvalue %30, %49[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %51 = llvm.insertvalue %39, %50[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%6 : i64)
  ^bb1(%52: i64):  // 2 preds: ^bb0, ^bb2
    %53 = llvm.icmp "slt" %52, %30 : i64
    llvm.cond_br %53, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %54 = llvm.getelementptr %44[%52] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %54 : !llvm.ptr<i64>
    %55 = llvm.add %52, %3  : i64
    llvm.br ^bb1(%55 : i64)
  ^bb3:  // pred: ^bb1
    %56 = llvm.mlir.constant(1 : index) : i64
    %57 = llvm.alloca %56 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %51, %57 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %58 = llvm.bitcast %57 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %61 = llvm.insertvalue %59, %60[0] : !llvm.struct<(i64, ptr<i8>)> 
    %62 = llvm.insertvalue %58, %61[1] : !llvm.struct<(i64, ptr<i8>)> 
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.mlir.null : !llvm.ptr<i64>
    %65 = llvm.getelementptr %64[%32] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %66 = llvm.ptrtoint %65 : !llvm.ptr<i64> to i64
    %67 = llvm.call @malloc(%66) : (i64) -> !llvm.ptr<i8>
    %68 = llvm.bitcast %67 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %69 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %71 = llvm.insertvalue %68, %70[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.insertvalue %32, %73[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.insertvalue %63, %74[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%6 : i64)
  ^bb4(%76: i64):  // 2 preds: ^bb3, ^bb5
    %77 = llvm.icmp "slt" %76, %32 : i64
    llvm.cond_br %77, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %78 = llvm.getelementptr %68[%76] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %78 : !llvm.ptr<i64>
    %79 = llvm.add %76, %3  : i64
    llvm.br ^bb4(%79 : i64)
  ^bb6:  // pred: ^bb4
    %80 = llvm.mlir.constant(1 : index) : i64
    %81 = llvm.alloca %80 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %75, %81 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %82 = llvm.bitcast %81 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %83 = llvm.mlir.constant(1 : index) : i64
    %84 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %85 = llvm.insertvalue %83, %84[0] : !llvm.struct<(i64, ptr<i8>)> 
    %86 = llvm.insertvalue %82, %85[1] : !llvm.struct<(i64, ptr<i8>)> 
    %87 = llvm.mlir.constant(1 : index) : i64
    %88 = llvm.mlir.null : !llvm.ptr<i64>
    %89 = llvm.getelementptr %88[%34] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %90 = llvm.ptrtoint %89 : !llvm.ptr<i64> to i64
    %91 = llvm.call @malloc(%90) : (i64) -> !llvm.ptr<i8>
    %92 = llvm.bitcast %91 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %93 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %94 = llvm.insertvalue %92, %93[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.mlir.constant(0 : index) : i64
    %97 = llvm.insertvalue %96, %95[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.insertvalue %34, %97[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.insertvalue %87, %98[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%6 : i64)
  ^bb7(%100: i64):  // 2 preds: ^bb6, ^bb8
    %101 = llvm.icmp "slt" %100, %34 : i64
    llvm.cond_br %101, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %102 = llvm.getelementptr %92[%100] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %102 : !llvm.ptr<i64>
    %103 = llvm.add %100, %3  : i64
    llvm.br ^bb7(%103 : i64)
  ^bb9:  // pred: ^bb7
    %104 = llvm.mlir.constant(1 : index) : i64
    %105 = llvm.alloca %104 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %99, %105 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %106 = llvm.bitcast %105 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %109 = llvm.insertvalue %107, %108[0] : !llvm.struct<(i64, ptr<i8>)> 
    %110 = llvm.insertvalue %106, %109[1] : !llvm.struct<(i64, ptr<i8>)> 
    %111 = llvm.mlir.constant(1 : index) : i64
    %112 = llvm.mlir.null : !llvm.ptr<i64>
    %113 = llvm.getelementptr %112[%36] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %114 = llvm.ptrtoint %113 : !llvm.ptr<i64> to i64
    %115 = llvm.call @malloc(%114) : (i64) -> !llvm.ptr<i8>
    %116 = llvm.bitcast %115 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %117 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %118 = llvm.insertvalue %116, %117[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %119 = llvm.insertvalue %116, %118[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %120 = llvm.mlir.constant(0 : index) : i64
    %121 = llvm.insertvalue %120, %119[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.insertvalue %36, %121[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.insertvalue %111, %122[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%6 : i64)
  ^bb10(%124: i64):  // 2 preds: ^bb9, ^bb11
    %125 = llvm.icmp "slt" %124, %36 : i64
    llvm.cond_br %125, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %126 = llvm.getelementptr %116[%124] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %126 : !llvm.ptr<i64>
    %127 = llvm.add %124, %3  : i64
    llvm.br ^bb10(%127 : i64)
  ^bb12:  // pred: ^bb10
    %128 = llvm.mlir.constant(1 : index) : i64
    %129 = llvm.alloca %128 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %123, %129 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %130 = llvm.bitcast %129 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %131 = llvm.mlir.constant(1 : index) : i64
    %132 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %133 = llvm.insertvalue %131, %132[0] : !llvm.struct<(i64, ptr<i8>)> 
    %134 = llvm.insertvalue %130, %133[1] : !llvm.struct<(i64, ptr<i8>)> 
    %135 = llvm.mlir.constant(1 : index) : i64
    %136 = llvm.mlir.null : !llvm.ptr<f64>
    %137 = llvm.getelementptr %136[%38] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %138 = llvm.ptrtoint %137 : !llvm.ptr<f64> to i64
    %139 = llvm.call @malloc(%138) : (i64) -> !llvm.ptr<i8>
    %140 = llvm.bitcast %139 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %141 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %142 = llvm.insertvalue %140, %141[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.insertvalue %140, %142[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.mlir.constant(0 : index) : i64
    %145 = llvm.insertvalue %144, %143[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.insertvalue %38, %145[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.insertvalue %135, %146[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%6 : i64)
  ^bb13(%148: i64):  // 2 preds: ^bb12, ^bb14
    %149 = llvm.icmp "slt" %148, %38 : i64
    llvm.cond_br %149, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %150 = llvm.getelementptr %140[%148] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %150 : !llvm.ptr<f64>
    %151 = llvm.add %148, %3  : i64
    llvm.br ^bb13(%151 : i64)
  ^bb15:  // pred: ^bb13
    %152 = llvm.mlir.constant(1 : index) : i64
    %153 = llvm.alloca %152 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %147, %153 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %154 = llvm.bitcast %153 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %155 = llvm.mlir.constant(1 : index) : i64
    %156 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %157 = llvm.insertvalue %155, %156[0] : !llvm.struct<(i64, ptr<i8>)> 
    %158 = llvm.insertvalue %154, %157[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%4, %6, %5, %6, %7, %59, %58, %83, %82, %107, %106, %131, %130, %155, %154, %4) {filename = "SPARSE_FILE_NAME1"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    llvm.call @comet_print_memref_i64(%59, %58) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%83, %82) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%107, %106) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%131, %130) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_f64(%155, %154) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @quick_sort(i64, !llvm.ptr<i8>, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_i64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
