module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(7 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.null : !llvm.ptr<i64>
    %10 = llvm.getelementptr %9[7] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %11 = llvm.ptrtoint %10 : !llvm.ptr<i64> to i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr<i8>
    %13 = llvm.bitcast %12 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %14 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.insertvalue %17, %16[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %7, %18[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %8, %19[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.mlir.constant(1 : index) : i64
    %22 = llvm.alloca %21 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %20, %22 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %23 = llvm.bitcast %22 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<(i64, ptr<i8>)> 
    %27 = llvm.insertvalue %23, %26[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_2D_f64(%4, %6, %5, %6, %6, %24, %23, %4) {filename = "SPARSE_FILE_NAME1"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %28 = llvm.getelementptr %13[%6] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %29 = llvm.load %28 : !llvm.ptr<i64>
    %30 = llvm.getelementptr %13[%3] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %31 = llvm.load %30 : !llvm.ptr<i64>
    %32 = llvm.getelementptr %13[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %33 = llvm.load %32 : !llvm.ptr<i64>
    %34 = llvm.getelementptr %13[%2] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %35 = llvm.load %34 : !llvm.ptr<i64>
    %36 = llvm.getelementptr %13[%1] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %37 = llvm.load %36 : !llvm.ptr<i64>
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.mlir.null : !llvm.ptr<i64>
    %40 = llvm.getelementptr %39[%29] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %41 = llvm.ptrtoint %40 : !llvm.ptr<i64> to i64
    %42 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr<i8>
    %43 = llvm.bitcast %42 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %44 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %43, %45[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.mlir.constant(0 : index) : i64
    %48 = llvm.insertvalue %47, %46[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.insertvalue %29, %48[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %50 = llvm.insertvalue %38, %49[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%6 : i64)
  ^bb1(%51: i64):  // 2 preds: ^bb0, ^bb2
    %52 = llvm.icmp "slt" %51, %29 : i64
    llvm.cond_br %52, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %53 = llvm.getelementptr %43[%51] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %53 : !llvm.ptr<i64>
    %54 = llvm.add %51, %3  : i64
    llvm.br ^bb1(%54 : i64)
  ^bb3:  // pred: ^bb1
    %55 = llvm.mlir.constant(1 : index) : i64
    %56 = llvm.alloca %55 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %50, %56 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %57 = llvm.bitcast %56 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %58 = llvm.mlir.constant(1 : index) : i64
    %59 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(i64, ptr<i8>)> 
    %61 = llvm.insertvalue %57, %60[1] : !llvm.struct<(i64, ptr<i8>)> 
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.mlir.null : !llvm.ptr<i64>
    %64 = llvm.getelementptr %63[%31] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %65 = llvm.ptrtoint %64 : !llvm.ptr<i64> to i64
    %66 = llvm.call @malloc(%65) : (i64) -> !llvm.ptr<i8>
    %67 = llvm.bitcast %66 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %68 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %69 = llvm.insertvalue %67, %68[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.insertvalue %67, %69[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %71 = llvm.mlir.constant(0 : index) : i64
    %72 = llvm.insertvalue %71, %70[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %73 = llvm.insertvalue %31, %72[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.insertvalue %62, %73[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%6 : i64)
  ^bb4(%75: i64):  // 2 preds: ^bb3, ^bb5
    %76 = llvm.icmp "slt" %75, %31 : i64
    llvm.cond_br %76, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %77 = llvm.getelementptr %67[%75] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %77 : !llvm.ptr<i64>
    %78 = llvm.add %75, %3  : i64
    llvm.br ^bb4(%78 : i64)
  ^bb6:  // pred: ^bb4
    %79 = llvm.mlir.constant(1 : index) : i64
    %80 = llvm.alloca %79 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %74, %80 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %81 = llvm.bitcast %80 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %82 = llvm.mlir.constant(1 : index) : i64
    %83 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(i64, ptr<i8>)> 
    %85 = llvm.insertvalue %81, %84[1] : !llvm.struct<(i64, ptr<i8>)> 
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.mlir.null : !llvm.ptr<i64>
    %88 = llvm.getelementptr %87[%33] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %89 = llvm.ptrtoint %88 : !llvm.ptr<i64> to i64
    %90 = llvm.call @malloc(%89) : (i64) -> !llvm.ptr<i8>
    %91 = llvm.bitcast %90 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %92 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %93 = llvm.insertvalue %91, %92[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %94 = llvm.insertvalue %91, %93[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.mlir.constant(0 : index) : i64
    %96 = llvm.insertvalue %95, %94[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %97 = llvm.insertvalue %33, %96[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.insertvalue %86, %97[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%6 : i64)
  ^bb7(%99: i64):  // 2 preds: ^bb6, ^bb8
    %100 = llvm.icmp "slt" %99, %33 : i64
    llvm.cond_br %100, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %101 = llvm.getelementptr %91[%99] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %101 : !llvm.ptr<i64>
    %102 = llvm.add %99, %3  : i64
    llvm.br ^bb7(%102 : i64)
  ^bb9:  // pred: ^bb7
    %103 = llvm.mlir.constant(1 : index) : i64
    %104 = llvm.alloca %103 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %98, %104 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %105 = llvm.bitcast %104 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %106 = llvm.mlir.constant(1 : index) : i64
    %107 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.struct<(i64, ptr<i8>)> 
    %109 = llvm.insertvalue %105, %108[1] : !llvm.struct<(i64, ptr<i8>)> 
    %110 = llvm.mlir.constant(1 : index) : i64
    %111 = llvm.mlir.null : !llvm.ptr<i64>
    %112 = llvm.getelementptr %111[%35] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %113 = llvm.ptrtoint %112 : !llvm.ptr<i64> to i64
    %114 = llvm.call @malloc(%113) : (i64) -> !llvm.ptr<i8>
    %115 = llvm.bitcast %114 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %116 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %117 = llvm.insertvalue %115, %116[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %118 = llvm.insertvalue %115, %117[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %119 = llvm.mlir.constant(0 : index) : i64
    %120 = llvm.insertvalue %119, %118[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %121 = llvm.insertvalue %35, %120[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.insertvalue %110, %121[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%6 : i64)
  ^bb10(%123: i64):  // 2 preds: ^bb9, ^bb11
    %124 = llvm.icmp "slt" %123, %35 : i64
    llvm.cond_br %124, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %125 = llvm.getelementptr %115[%123] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %6, %125 : !llvm.ptr<i64>
    %126 = llvm.add %123, %3  : i64
    llvm.br ^bb10(%126 : i64)
  ^bb12:  // pred: ^bb10
    %127 = llvm.mlir.constant(1 : index) : i64
    %128 = llvm.alloca %127 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %122, %128 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %129 = llvm.bitcast %128 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %130 = llvm.mlir.constant(1 : index) : i64
    %131 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %132 = llvm.insertvalue %130, %131[0] : !llvm.struct<(i64, ptr<i8>)> 
    %133 = llvm.insertvalue %129, %132[1] : !llvm.struct<(i64, ptr<i8>)> 
    %134 = llvm.mlir.constant(1 : index) : i64
    %135 = llvm.mlir.null : !llvm.ptr<f64>
    %136 = llvm.getelementptr %135[%37] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %137 = llvm.ptrtoint %136 : !llvm.ptr<f64> to i64
    %138 = llvm.call @malloc(%137) : (i64) -> !llvm.ptr<i8>
    %139 = llvm.bitcast %138 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %140 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %141 = llvm.insertvalue %139, %140[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.insertvalue %139, %141[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.mlir.constant(0 : index) : i64
    %144 = llvm.insertvalue %143, %142[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.insertvalue %37, %144[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.insertvalue %134, %145[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%6 : i64)
  ^bb13(%147: i64):  // 2 preds: ^bb12, ^bb14
    %148 = llvm.icmp "slt" %147, %37 : i64
    llvm.cond_br %148, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %149 = llvm.getelementptr %139[%147] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %149 : !llvm.ptr<f64>
    %150 = llvm.add %147, %3  : i64
    llvm.br ^bb13(%150 : i64)
  ^bb15:  // pred: ^bb13
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.alloca %151 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %146, %152 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %153 = llvm.bitcast %152 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %154 = llvm.mlir.constant(1 : index) : i64
    %155 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %156 = llvm.insertvalue %154, %155[0] : !llvm.struct<(i64, ptr<i8>)> 
    %157 = llvm.insertvalue %153, %156[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%4, %6, %5, %58, %57, %82, %81, %106, %105, %130, %129, %154, %153, %4) {filename = "SPARSE_FILE_NAME1"} : (i32, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    llvm.call @comet_print_memref_i64(%58, %57) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%82, %81) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%106, %105) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%130, %129) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_f64(%154, %153) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @quick_sort(i64, !llvm.ptr<i8>, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_i64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
