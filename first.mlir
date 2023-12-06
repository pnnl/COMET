module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(32 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(13 : index) : i64
    %6 = llvm.mlir.constant(2 : index) : i64
    %7 = llvm.mlir.constant(3 : index) : i64
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(5 : index) : i64
    %11 = llvm.mlir.constant(6 : index) : i64
    %12 = llvm.mlir.constant(7 : index) : i64
    %13 = llvm.mlir.constant(8 : index) : i64
    %14 = llvm.mlir.constant(9 : index) : i64
    %15 = llvm.mlir.constant(10 : index) : i64
    %16 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %17 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = llvm.mlir.null : !llvm.ptr<i64>
    %22 = llvm.getelementptr %21[13] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %23 = llvm.ptrtoint %22 : !llvm.ptr<i64> to i64
    %24 = llvm.call @malloc(%23) : (i64) -> !llvm.ptr<i8>
    %25 = llvm.bitcast %24 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %25, %27[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %3, %28[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %5, %29[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.insertvalue %4, %30[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %31, %32 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %33 = llvm.bitcast %32 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %34 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %35 = llvm.insertvalue %4, %34[0] : !llvm.struct<(i64, ptr<i8>)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_2D_f64(%8, %20, %20, %19, %20, %4, %33, %9) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %37 = llvm.load %25 : !llvm.ptr<i64>
    %38 = llvm.getelementptr %25[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %39 = llvm.load %38 : !llvm.ptr<i64>
    %40 = llvm.getelementptr %25[2] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %41 = llvm.load %40 : !llvm.ptr<i64>
    %42 = llvm.getelementptr %25[3] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %43 = llvm.load %42 : !llvm.ptr<i64>
    %44 = llvm.getelementptr %25[4] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %45 = llvm.load %44 : !llvm.ptr<i64>
    %46 = llvm.getelementptr %25[5] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %47 = llvm.load %46 : !llvm.ptr<i64>
    %48 = llvm.getelementptr %25[6] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %49 = llvm.load %48 : !llvm.ptr<i64>
    %50 = llvm.getelementptr %25[7] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %51 = llvm.load %50 : !llvm.ptr<i64>
    %52 = llvm.getelementptr %25[8] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %53 = llvm.load %52 : !llvm.ptr<i64>
    %54 = llvm.getelementptr %25[9] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %55 = llvm.load %54 : !llvm.ptr<i64>
    %56 = llvm.getelementptr %25[10] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %57 = llvm.load %56 : !llvm.ptr<i64>
    %58 = llvm.mlir.null : !llvm.ptr<i64>
    %59 = llvm.getelementptr %58[%37] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %60 = llvm.ptrtoint %59 : !llvm.ptr<i64> to i64
    %61 = llvm.call @malloc(%60) : (i64) -> !llvm.ptr<i8>
    %62 = llvm.bitcast %61 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %63 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %3, %65[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %37, %66[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.insertvalue %4, %67[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%20 : i64)
  ^bb1(%69: i64):  // 2 preds: ^bb0, ^bb2
    %70 = llvm.icmp "slt" %69, %37 : i64
    llvm.cond_br %70, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %71 = llvm.getelementptr %62[%69] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %71 : !llvm.ptr<i64>
    %72 = llvm.add %69, %19  : i64
    llvm.br ^bb1(%72 : i64)
  ^bb3:  // pred: ^bb1
    %73 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %68, %73 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %74 = llvm.bitcast %73 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %75 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %76 = llvm.insertvalue %4, %75[0] : !llvm.struct<(i64, ptr<i8>)> 
    %77 = llvm.insertvalue %74, %76[1] : !llvm.struct<(i64, ptr<i8>)> 
    %78 = llvm.mlir.null : !llvm.ptr<i64>
    %79 = llvm.getelementptr %78[%39] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %80 = llvm.ptrtoint %79 : !llvm.ptr<i64> to i64
    %81 = llvm.call @malloc(%80) : (i64) -> !llvm.ptr<i8>
    %82 = llvm.bitcast %81 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %83 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.insertvalue %82, %84[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %3, %85[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.insertvalue %39, %86[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.insertvalue %4, %87[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%20 : i64)
  ^bb4(%89: i64):  // 2 preds: ^bb3, ^bb5
    %90 = llvm.icmp "slt" %89, %39 : i64
    llvm.cond_br %90, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %91 = llvm.getelementptr %82[%89] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %91 : !llvm.ptr<i64>
    %92 = llvm.add %89, %19  : i64
    llvm.br ^bb4(%92 : i64)
  ^bb6:  // pred: ^bb4
    %93 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %88, %93 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %94 = llvm.bitcast %93 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %95 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %96 = llvm.insertvalue %4, %95[0] : !llvm.struct<(i64, ptr<i8>)> 
    %97 = llvm.insertvalue %94, %96[1] : !llvm.struct<(i64, ptr<i8>)> 
    %98 = llvm.mlir.null : !llvm.ptr<i64>
    %99 = llvm.getelementptr %98[%41] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %100 = llvm.ptrtoint %99 : !llvm.ptr<i64> to i64
    %101 = llvm.call @malloc(%100) : (i64) -> !llvm.ptr<i8>
    %102 = llvm.bitcast %101 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %103 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.insertvalue %3, %105[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.insertvalue %41, %106[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %108 = llvm.insertvalue %4, %107[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%20 : i64)
  ^bb7(%109: i64):  // 2 preds: ^bb6, ^bb8
    %110 = llvm.icmp "slt" %109, %41 : i64
    llvm.cond_br %110, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %111 = llvm.getelementptr %102[%109] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %111 : !llvm.ptr<i64>
    %112 = llvm.add %109, %19  : i64
    llvm.br ^bb7(%112 : i64)
  ^bb9:  // pred: ^bb7
    %113 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %108, %113 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %114 = llvm.bitcast %113 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %115 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %116 = llvm.insertvalue %4, %115[0] : !llvm.struct<(i64, ptr<i8>)> 
    %117 = llvm.insertvalue %114, %116[1] : !llvm.struct<(i64, ptr<i8>)> 
    %118 = llvm.mlir.null : !llvm.ptr<i64>
    %119 = llvm.getelementptr %118[%43] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %120 = llvm.ptrtoint %119 : !llvm.ptr<i64> to i64
    %121 = llvm.call @malloc(%120) : (i64) -> !llvm.ptr<i8>
    %122 = llvm.bitcast %121 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %123 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %124 = llvm.insertvalue %122, %123[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.insertvalue %122, %124[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.insertvalue %3, %125[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.insertvalue %43, %126[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.insertvalue %4, %127[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%20 : i64)
  ^bb10(%129: i64):  // 2 preds: ^bb9, ^bb11
    %130 = llvm.icmp "slt" %129, %43 : i64
    llvm.cond_br %130, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %131 = llvm.getelementptr %122[%129] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %131 : !llvm.ptr<i64>
    %132 = llvm.add %129, %19  : i64
    llvm.br ^bb10(%132 : i64)
  ^bb12:  // pred: ^bb10
    %133 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %128, %133 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %134 = llvm.bitcast %133 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %135 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %136 = llvm.insertvalue %4, %135[0] : !llvm.struct<(i64, ptr<i8>)> 
    %137 = llvm.insertvalue %134, %136[1] : !llvm.struct<(i64, ptr<i8>)> 
    %138 = llvm.mlir.null : !llvm.ptr<i64>
    %139 = llvm.getelementptr %138[%45] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %140 = llvm.ptrtoint %139 : !llvm.ptr<i64> to i64
    %141 = llvm.call @malloc(%140) : (i64) -> !llvm.ptr<i8>
    %142 = llvm.bitcast %141 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %143 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %144 = llvm.insertvalue %142, %143[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.insertvalue %142, %144[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.insertvalue %3, %145[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.insertvalue %45, %146[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %148 = llvm.insertvalue %4, %147[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%20 : i64)
  ^bb13(%149: i64):  // 2 preds: ^bb12, ^bb14
    %150 = llvm.icmp "slt" %149, %45 : i64
    llvm.cond_br %150, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %151 = llvm.getelementptr %142[%149] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %151 : !llvm.ptr<i64>
    %152 = llvm.add %149, %19  : i64
    llvm.br ^bb13(%152 : i64)
  ^bb15:  // pred: ^bb13
    %153 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %148, %153 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %154 = llvm.bitcast %153 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %155 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %156 = llvm.insertvalue %4, %155[0] : !llvm.struct<(i64, ptr<i8>)> 
    %157 = llvm.insertvalue %154, %156[1] : !llvm.struct<(i64, ptr<i8>)> 
    %158 = llvm.mlir.null : !llvm.ptr<i64>
    %159 = llvm.getelementptr %158[%47] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %160 = llvm.ptrtoint %159 : !llvm.ptr<i64> to i64
    %161 = llvm.call @malloc(%160) : (i64) -> !llvm.ptr<i8>
    %162 = llvm.bitcast %161 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %163 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %164 = llvm.insertvalue %162, %163[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %165 = llvm.insertvalue %162, %164[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.insertvalue %3, %165[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.insertvalue %47, %166[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %168 = llvm.insertvalue %4, %167[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%20 : i64)
  ^bb16(%169: i64):  // 2 preds: ^bb15, ^bb17
    %170 = llvm.icmp "slt" %169, %47 : i64
    llvm.cond_br %170, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %171 = llvm.getelementptr %162[%169] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %171 : !llvm.ptr<i64>
    %172 = llvm.add %169, %19  : i64
    llvm.br ^bb16(%172 : i64)
  ^bb18:  // pred: ^bb16
    %173 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %168, %173 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %174 = llvm.bitcast %173 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %175 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %176 = llvm.insertvalue %4, %175[0] : !llvm.struct<(i64, ptr<i8>)> 
    %177 = llvm.insertvalue %174, %176[1] : !llvm.struct<(i64, ptr<i8>)> 
    %178 = llvm.mlir.null : !llvm.ptr<i64>
    %179 = llvm.getelementptr %178[%49] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %180 = llvm.ptrtoint %179 : !llvm.ptr<i64> to i64
    %181 = llvm.call @malloc(%180) : (i64) -> !llvm.ptr<i8>
    %182 = llvm.bitcast %181 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %183 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %184 = llvm.insertvalue %182, %183[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %185 = llvm.insertvalue %182, %184[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %186 = llvm.insertvalue %3, %185[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %187 = llvm.insertvalue %49, %186[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %188 = llvm.insertvalue %4, %187[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%20 : i64)
  ^bb19(%189: i64):  // 2 preds: ^bb18, ^bb20
    %190 = llvm.icmp "slt" %189, %49 : i64
    llvm.cond_br %190, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %191 = llvm.getelementptr %182[%189] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %191 : !llvm.ptr<i64>
    %192 = llvm.add %189, %19  : i64
    llvm.br ^bb19(%192 : i64)
  ^bb21:  // pred: ^bb19
    %193 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %188, %193 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %194 = llvm.bitcast %193 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %195 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %196 = llvm.insertvalue %4, %195[0] : !llvm.struct<(i64, ptr<i8>)> 
    %197 = llvm.insertvalue %194, %196[1] : !llvm.struct<(i64, ptr<i8>)> 
    %198 = llvm.mlir.null : !llvm.ptr<i64>
    %199 = llvm.getelementptr %198[%51] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %200 = llvm.ptrtoint %199 : !llvm.ptr<i64> to i64
    %201 = llvm.call @malloc(%200) : (i64) -> !llvm.ptr<i8>
    %202 = llvm.bitcast %201 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %203 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %204 = llvm.insertvalue %202, %203[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %205 = llvm.insertvalue %202, %204[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %206 = llvm.insertvalue %3, %205[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %207 = llvm.insertvalue %51, %206[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %208 = llvm.insertvalue %4, %207[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%20 : i64)
  ^bb22(%209: i64):  // 2 preds: ^bb21, ^bb23
    %210 = llvm.icmp "slt" %209, %51 : i64
    llvm.cond_br %210, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %211 = llvm.getelementptr %202[%209] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %20, %211 : !llvm.ptr<i64>
    %212 = llvm.add %209, %19  : i64
    llvm.br ^bb22(%212 : i64)
  ^bb24:  // pred: ^bb22
    %213 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %208, %213 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %214 = llvm.bitcast %213 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %215 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %216 = llvm.insertvalue %4, %215[0] : !llvm.struct<(i64, ptr<i8>)> 
    %217 = llvm.insertvalue %214, %216[1] : !llvm.struct<(i64, ptr<i8>)> 
    %218 = llvm.mlir.null : !llvm.ptr<f64>
    %219 = llvm.getelementptr %218[%53] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %220 = llvm.ptrtoint %219 : !llvm.ptr<f64> to i64
    %221 = llvm.call @malloc(%220) : (i64) -> !llvm.ptr<i8>
    %222 = llvm.bitcast %221 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %223 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %224 = llvm.insertvalue %222, %223[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %225 = llvm.insertvalue %222, %224[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %226 = llvm.insertvalue %3, %225[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %227 = llvm.insertvalue %53, %226[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %228 = llvm.insertvalue %4, %227[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%20 : i64)
  ^bb25(%229: i64):  // 2 preds: ^bb24, ^bb26
    %230 = llvm.icmp "slt" %229, %53 : i64
    llvm.cond_br %230, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %231 = llvm.getelementptr %222[%229] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %16, %231 : !llvm.ptr<f64>
    %232 = llvm.add %229, %19  : i64
    llvm.br ^bb25(%232 : i64)
  ^bb27:  // pred: ^bb25
    %233 = llvm.alloca %4 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %228, %233 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %234 = llvm.bitcast %233 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %235 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %236 = llvm.insertvalue %4, %235[0] : !llvm.struct<(i64, ptr<i8>)> 
    %237 = llvm.insertvalue %234, %236[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%8, %20, %20, %19, %20, %4, %74, %4, %94, %4, %114, %4, %134, %4, %154, %4, %174, %4, %194, %4, %214, %4, %234, %9) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    %238 = llvm.mul %57, %2  : i64
    %239 = llvm.mlir.null : !llvm.ptr<f64>
    %240 = llvm.getelementptr %239[%238] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %241 = llvm.ptrtoint %240 : !llvm.ptr<f64> to i64
    %242 = llvm.add %241, %1  : i64
    %243 = llvm.call @malloc(%242) : (i64) -> !llvm.ptr<i8>
    %244 = llvm.bitcast %243 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %245 = llvm.ptrtoint %244 : !llvm.ptr<f64> to i64
    %246 = llvm.sub %1, %4  : i64
    %247 = llvm.add %245, %246  : i64
    %248 = llvm.urem %247, %1  : i64
    %249 = llvm.sub %247, %248  : i64
    %250 = llvm.inttoptr %249 : i64 to !llvm.ptr<f64>
    %251 = llvm.mul %55, %2  : i64
    %252 = llvm.mlir.null : !llvm.ptr<f64>
    %253 = llvm.getelementptr %252[%251] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %254 = llvm.ptrtoint %253 : !llvm.ptr<f64> to i64
    %255 = llvm.add %254, %1  : i64
    %256 = llvm.call @malloc(%255) : (i64) -> !llvm.ptr<i8>
    %257 = llvm.bitcast %256 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %258 = llvm.ptrtoint %257 : !llvm.ptr<f64> to i64
    %259 = llvm.sub %1, %4  : i64
    %260 = llvm.add %258, %259  : i64
    %261 = llvm.urem %260, %1  : i64
    %262 = llvm.sub %260, %261  : i64
    %263 = llvm.inttoptr %262 : i64 to !llvm.ptr<f64>
    %264 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %265 = llvm.insertvalue %257, %264[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %266 = llvm.insertvalue %263, %265[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %267 = llvm.insertvalue %3, %266[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %268 = llvm.insertvalue %55, %267[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %269 = llvm.insertvalue %2, %268[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %270 = llvm.insertvalue %2, %269[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %271 = llvm.insertvalue %4, %270[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @comet_print_memref_i64(%4, %74) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%4, %94) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%4, %114) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%4, %134) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%4, %154) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%4, %174) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%4, %194) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%4, %214) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_f64(%4, %234) : (i64, !llvm.ptr<i8>) -> ()
    llvm.br ^bb28(%20 : i64)
  ^bb28(%272: i64):  // 2 preds: ^bb27, ^bb31
    %273 = llvm.icmp "slt" %272, %57 : i64
    llvm.cond_br %273, ^bb29(%20 : i64), ^bb32(%20 : i64)
  ^bb29(%274: i64):  // 2 preds: ^bb28, ^bb30
    %275 = llvm.icmp "slt" %274, %18 : i64
    llvm.cond_br %275, ^bb30, ^bb31
  ^bb30:  // pred: ^bb29
    %276 = llvm.mul %272, %2  : i64
    %277 = llvm.add %276, %274  : i64
    %278 = llvm.getelementptr %250[%277] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %17, %278 : !llvm.ptr<f64>
    %279 = llvm.add %274, %19  : i64
    llvm.br ^bb29(%279 : i64)
  ^bb31:  // pred: ^bb29
    %280 = llvm.add %272, %19  : i64
    llvm.br ^bb28(%280 : i64)
  ^bb32(%281: i64):  // 2 preds: ^bb28, ^bb35
    %282 = llvm.icmp "slt" %281, %55 : i64
    llvm.cond_br %282, ^bb33(%20 : i64), ^bb36
  ^bb33(%283: i64):  // 2 preds: ^bb32, ^bb34
    %284 = llvm.icmp "slt" %283, %18 : i64
    llvm.cond_br %284, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %285 = llvm.mul %281, %2  : i64
    %286 = llvm.add %285, %283  : i64
    %287 = llvm.getelementptr %263[%286] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %16, %287 : !llvm.ptr<f64>
    %288 = llvm.add %283, %19  : i64
    llvm.br ^bb33(%288 : i64)
  ^bb35:  // pred: ^bb33
    %289 = llvm.add %281, %19  : i64
    llvm.br ^bb32(%289 : i64)
  ^bb36:  // pred: ^bb32
    %290 = llvm.load %62 : !llvm.ptr<i64>
    llvm.br ^bb37(%20 : i64)
  ^bb37(%291: i64):  // 2 preds: ^bb36, ^bb50
    %292 = llvm.icmp "slt" %291, %290 : i64
    llvm.cond_br %292, ^bb38, ^bb51
  ^bb38:  // pred: ^bb37
    %293 = llvm.load %102 : !llvm.ptr<i64>
    llvm.br ^bb39(%20 : i64)
  ^bb39(%294: i64):  // 2 preds: ^bb38, ^bb49
    %295 = llvm.icmp "slt" %294, %293 : i64
    llvm.cond_br %295, ^bb40, ^bb50
  ^bb40:  // pred: ^bb39
    %296 = llvm.load %102 : !llvm.ptr<i64>
    %297 = llvm.mul %291, %296  : i64
    %298 = llvm.add %297, %294  : i64
    %299 = llvm.add %291, %19  : i64
    %300 = llvm.getelementptr %142[%291] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %301 = llvm.load %300 : !llvm.ptr<i64>
    %302 = llvm.getelementptr %142[%299] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %303 = llvm.load %302 : !llvm.ptr<i64>
    llvm.br ^bb41(%301 : i64)
  ^bb41(%304: i64):  // 2 preds: ^bb40, ^bb48
    %305 = llvm.icmp "slt" %304, %303 : i64
    llvm.cond_br %305, ^bb42, ^bb49
  ^bb42:  // pred: ^bb41
    %306 = llvm.getelementptr %162[%304] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %307 = llvm.load %306 : !llvm.ptr<i64>
    %308 = llvm.load %182 : !llvm.ptr<i64>
    llvm.br ^bb43(%20 : i64)
  ^bb43(%309: i64):  // 2 preds: ^bb42, ^bb47
    %310 = llvm.icmp "slt" %309, %308 : i64
    llvm.cond_br %310, ^bb44, ^bb48
  ^bb44:  // pred: ^bb43
    %311 = llvm.getelementptr %182[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %312 = llvm.load %311 : !llvm.ptr<i64>
    %313 = llvm.mul %307, %312  : i64
    %314 = llvm.add %313, %309  : i64
    llvm.br ^bb45(%20 : i64)
  ^bb45(%315: i64):  // 2 preds: ^bb44, ^bb46
    %316 = llvm.icmp "slt" %315, %18 : i64
    llvm.cond_br %316, ^bb46, ^bb47
  ^bb46:  // pred: ^bb45
    %317 = llvm.load %102 : !llvm.ptr<i64>
    %318 = llvm.load %182 : !llvm.ptr<i64>
    %319 = llvm.mul %317, %318  : i64
    %320 = llvm.mul %304, %319  : i64
    %321 = llvm.mul %294, %318  : i64
    %322 = llvm.add %320, %321  : i64
    %323 = llvm.add %322, %309  : i64
    %324 = llvm.getelementptr %222[%323] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %325 = llvm.load %324 : !llvm.ptr<f64>
    %326 = llvm.mul %314, %2  : i64
    %327 = llvm.add %326, %315  : i64
    %328 = llvm.getelementptr %250[%327] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %329 = llvm.load %328 : !llvm.ptr<f64>
    %330 = llvm.mul %298, %2  : i64
    %331 = llvm.add %330, %315  : i64
    %332 = llvm.getelementptr %263[%331] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %333 = llvm.load %332 : !llvm.ptr<f64>
    %334 = llvm.fmul %325, %329  : f64
    %335 = llvm.fadd %333, %334  : f64
    %336 = llvm.mul %298, %2  : i64
    %337 = llvm.add %336, %315  : i64
    %338 = llvm.getelementptr %263[%337] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %335, %338 : !llvm.ptr<f64>
    %339 = llvm.add %315, %19  : i64
    llvm.br ^bb45(%339 : i64)
  ^bb47:  // pred: ^bb45
    %340 = llvm.add %309, %19  : i64
    llvm.br ^bb43(%340 : i64)
  ^bb48:  // pred: ^bb43
    %341 = llvm.add %304, %19  : i64
    llvm.br ^bb41(%341 : i64)
  ^bb49:  // pred: ^bb41
    %342 = llvm.add %294, %19  : i64
    llvm.br ^bb39(%342 : i64)
  ^bb50:  // pred: ^bb39
    %343 = llvm.add %291, %19  : i64
    llvm.br ^bb37(%343 : i64)
  ^bb51:  // pred: ^bb37
    %344 = llvm.alloca %4 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.store %271, %344 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
    %345 = llvm.bitcast %344 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
    %346 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %347 = llvm.insertvalue %0, %346[0] : !llvm.struct<(i64, ptr<i8>)> 
    %348 = llvm.insertvalue %345, %347[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @comet_print_memref_f64(%0, %345) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @comet_sort_index(i64, !llvm.ptr<i8>, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_i64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
