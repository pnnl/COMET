module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(32 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(13 : index) : i64
    %6 = llvm.mlir.constant(-1 : index) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(3 : index) : i64
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(5 : index) : i64
    %12 = llvm.mlir.constant(6 : index) : i64
    %13 = llvm.mlir.constant(7 : index) : i64
    %14 = llvm.mlir.constant(8 : index) : i64
    %15 = llvm.mlir.constant(9 : index) : i64
    %16 = llvm.mlir.constant(10 : index) : i64
    %17 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %18 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %19 = llvm.mlir.constant(4 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.mlir.null : !llvm.ptr<i64>
    %23 = llvm.getelementptr %22[13] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<i64> to i64
    %25 = llvm.call @malloc(%24) : (i64) -> !llvm.ptr<i8>
    %26 = llvm.bitcast %25 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %27 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %26, %27[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %3, %29[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.insertvalue %5, %30[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %4, %31[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %32, %33 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %34 = llvm.bitcast %33 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %35 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %36 = llvm.insertvalue %4, %35[0] : !llvm.struct<(i64, ptr<i8>)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_2D_f64(%9, %21, %6, %20, %6, %4, %34, %10) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %38 = llvm.load %26 : !llvm.ptr<i64>
    %39 = llvm.getelementptr %26[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %40 = llvm.load %39 : !llvm.ptr<i64>
    %41 = llvm.getelementptr %26[2] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %42 = llvm.load %41 : !llvm.ptr<i64>
    %43 = llvm.getelementptr %26[3] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %44 = llvm.load %43 : !llvm.ptr<i64>
    %45 = llvm.getelementptr %26[4] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %46 = llvm.load %45 : !llvm.ptr<i64>
    %47 = llvm.getelementptr %26[5] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %48 = llvm.load %47 : !llvm.ptr<i64>
    %49 = llvm.getelementptr %26[6] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %50 = llvm.load %49 : !llvm.ptr<i64>
    %51 = llvm.getelementptr %26[7] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %52 = llvm.load %51 : !llvm.ptr<i64>
    %53 = llvm.getelementptr %26[8] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %54 = llvm.load %53 : !llvm.ptr<i64>
    %55 = llvm.getelementptr %26[9] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %56 = llvm.load %55 : !llvm.ptr<i64>
    %57 = llvm.getelementptr %26[10] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %58 = llvm.load %57 : !llvm.ptr<i64>
    %59 = llvm.mlir.null : !llvm.ptr<i64>
    %60 = llvm.getelementptr %59[%38] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %61 = llvm.ptrtoint %60 : !llvm.ptr<i64> to i64
    %62 = llvm.call @malloc(%61) : (i64) -> !llvm.ptr<i8>
    %63 = llvm.bitcast %62 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %64 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %63, %65[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %3, %66[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.insertvalue %38, %67[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.insertvalue %4, %68[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%21 : i64)
  ^bb1(%70: i64):  // 2 preds: ^bb0, ^bb2
    %71 = llvm.icmp "slt" %70, %38 : i64
    llvm.cond_br %71, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %72 = llvm.getelementptr %63[%70] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %72 : !llvm.ptr<i64>
    %73 = llvm.add %70, %20  : i64
    llvm.br ^bb1(%73 : i64)
  ^bb3:  // pred: ^bb1
    %74 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %69, %74 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %75 = llvm.bitcast %74 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %76 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %77 = llvm.insertvalue %4, %76[0] : !llvm.struct<(i64, ptr<i8>)> 
    %78 = llvm.insertvalue %75, %77[1] : !llvm.struct<(i64, ptr<i8>)> 
    %79 = llvm.mlir.null : !llvm.ptr<i64>
    %80 = llvm.getelementptr %79[%40] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %81 = llvm.ptrtoint %80 : !llvm.ptr<i64> to i64
    %82 = llvm.call @malloc(%81) : (i64) -> !llvm.ptr<i8>
    %83 = llvm.bitcast %82 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %84 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %85 = llvm.insertvalue %83, %84[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %83, %85[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.insertvalue %3, %86[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.insertvalue %40, %87[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.insertvalue %4, %88[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%21 : i64)
  ^bb4(%90: i64):  // 2 preds: ^bb3, ^bb5
    %91 = llvm.icmp "slt" %90, %40 : i64
    llvm.cond_br %91, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %92 = llvm.getelementptr %83[%90] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %92 : !llvm.ptr<i64>
    %93 = llvm.add %90, %20  : i64
    llvm.br ^bb4(%93 : i64)
  ^bb6:  // pred: ^bb4
    %94 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %89, %94 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %95 = llvm.bitcast %94 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %96 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %97 = llvm.insertvalue %4, %96[0] : !llvm.struct<(i64, ptr<i8>)> 
    %98 = llvm.insertvalue %95, %97[1] : !llvm.struct<(i64, ptr<i8>)> 
    %99 = llvm.mlir.null : !llvm.ptr<i64>
    %100 = llvm.getelementptr %99[%42] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %101 = llvm.ptrtoint %100 : !llvm.ptr<i64> to i64
    %102 = llvm.call @malloc(%101) : (i64) -> !llvm.ptr<i8>
    %103 = llvm.bitcast %102 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %104 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %105 = llvm.insertvalue %103, %104[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.insertvalue %103, %105[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.insertvalue %3, %106[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %108 = llvm.insertvalue %42, %107[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %109 = llvm.insertvalue %4, %108[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%21 : i64)
  ^bb7(%110: i64):  // 2 preds: ^bb6, ^bb8
    %111 = llvm.icmp "slt" %110, %42 : i64
    llvm.cond_br %111, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %112 = llvm.getelementptr %103[%110] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %112 : !llvm.ptr<i64>
    %113 = llvm.add %110, %20  : i64
    llvm.br ^bb7(%113 : i64)
  ^bb9:  // pred: ^bb7
    %114 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %109, %114 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %115 = llvm.bitcast %114 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %116 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %117 = llvm.insertvalue %4, %116[0] : !llvm.struct<(i64, ptr<i8>)> 
    %118 = llvm.insertvalue %115, %117[1] : !llvm.struct<(i64, ptr<i8>)> 
    %119 = llvm.mlir.null : !llvm.ptr<i64>
    %120 = llvm.getelementptr %119[%44] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %121 = llvm.ptrtoint %120 : !llvm.ptr<i64> to i64
    %122 = llvm.call @malloc(%121) : (i64) -> !llvm.ptr<i8>
    %123 = llvm.bitcast %122 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %124 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %125 = llvm.insertvalue %123, %124[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.insertvalue %123, %125[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.insertvalue %3, %126[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.insertvalue %44, %127[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %129 = llvm.insertvalue %4, %128[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%21 : i64)
  ^bb10(%130: i64):  // 2 preds: ^bb9, ^bb11
    %131 = llvm.icmp "slt" %130, %44 : i64
    llvm.cond_br %131, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %132 = llvm.getelementptr %123[%130] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %132 : !llvm.ptr<i64>
    %133 = llvm.add %130, %20  : i64
    llvm.br ^bb10(%133 : i64)
  ^bb12:  // pred: ^bb10
    %134 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %129, %134 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %135 = llvm.bitcast %134 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %136 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %137 = llvm.insertvalue %4, %136[0] : !llvm.struct<(i64, ptr<i8>)> 
    %138 = llvm.insertvalue %135, %137[1] : !llvm.struct<(i64, ptr<i8>)> 
    %139 = llvm.mlir.null : !llvm.ptr<i64>
    %140 = llvm.getelementptr %139[%46] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %141 = llvm.ptrtoint %140 : !llvm.ptr<i64> to i64
    %142 = llvm.call @malloc(%141) : (i64) -> !llvm.ptr<i8>
    %143 = llvm.bitcast %142 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %144 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %145 = llvm.insertvalue %143, %144[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.insertvalue %143, %145[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.insertvalue %3, %146[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %148 = llvm.insertvalue %46, %147[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %149 = llvm.insertvalue %4, %148[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%21 : i64)
  ^bb13(%150: i64):  // 2 preds: ^bb12, ^bb14
    %151 = llvm.icmp "slt" %150, %46 : i64
    llvm.cond_br %151, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %152 = llvm.getelementptr %143[%150] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %152 : !llvm.ptr<i64>
    %153 = llvm.add %150, %20  : i64
    llvm.br ^bb13(%153 : i64)
  ^bb15:  // pred: ^bb13
    %154 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %149, %154 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %155 = llvm.bitcast %154 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %156 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %157 = llvm.insertvalue %4, %156[0] : !llvm.struct<(i64, ptr<i8>)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(i64, ptr<i8>)> 
    %159 = llvm.mlir.null : !llvm.ptr<i64>
    %160 = llvm.getelementptr %159[%48] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %161 = llvm.ptrtoint %160 : !llvm.ptr<i64> to i64
    %162 = llvm.call @malloc(%161) : (i64) -> !llvm.ptr<i8>
    %163 = llvm.bitcast %162 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %164 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %165 = llvm.insertvalue %163, %164[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.insertvalue %163, %165[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.insertvalue %3, %166[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %168 = llvm.insertvalue %48, %167[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %169 = llvm.insertvalue %4, %168[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%21 : i64)
  ^bb16(%170: i64):  // 2 preds: ^bb15, ^bb17
    %171 = llvm.icmp "slt" %170, %48 : i64
    llvm.cond_br %171, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %172 = llvm.getelementptr %163[%170] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %172 : !llvm.ptr<i64>
    %173 = llvm.add %170, %20  : i64
    llvm.br ^bb16(%173 : i64)
  ^bb18:  // pred: ^bb16
    %174 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %169, %174 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %175 = llvm.bitcast %174 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %176 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %177 = llvm.insertvalue %4, %176[0] : !llvm.struct<(i64, ptr<i8>)> 
    %178 = llvm.insertvalue %175, %177[1] : !llvm.struct<(i64, ptr<i8>)> 
    %179 = llvm.mlir.null : !llvm.ptr<i64>
    %180 = llvm.getelementptr %179[%50] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %181 = llvm.ptrtoint %180 : !llvm.ptr<i64> to i64
    %182 = llvm.call @malloc(%181) : (i64) -> !llvm.ptr<i8>
    %183 = llvm.bitcast %182 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %184 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %185 = llvm.insertvalue %183, %184[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %186 = llvm.insertvalue %183, %185[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %187 = llvm.insertvalue %3, %186[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %188 = llvm.insertvalue %50, %187[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %189 = llvm.insertvalue %4, %188[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%21 : i64)
  ^bb19(%190: i64):  // 2 preds: ^bb18, ^bb20
    %191 = llvm.icmp "slt" %190, %50 : i64
    llvm.cond_br %191, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %192 = llvm.getelementptr %183[%190] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %192 : !llvm.ptr<i64>
    %193 = llvm.add %190, %20  : i64
    llvm.br ^bb19(%193 : i64)
  ^bb21:  // pred: ^bb19
    %194 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %189, %194 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %195 = llvm.bitcast %194 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %196 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %197 = llvm.insertvalue %4, %196[0] : !llvm.struct<(i64, ptr<i8>)> 
    %198 = llvm.insertvalue %195, %197[1] : !llvm.struct<(i64, ptr<i8>)> 
    %199 = llvm.mlir.null : !llvm.ptr<i64>
    %200 = llvm.getelementptr %199[%52] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %201 = llvm.ptrtoint %200 : !llvm.ptr<i64> to i64
    %202 = llvm.call @malloc(%201) : (i64) -> !llvm.ptr<i8>
    %203 = llvm.bitcast %202 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %204 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %205 = llvm.insertvalue %203, %204[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %206 = llvm.insertvalue %203, %205[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %207 = llvm.insertvalue %3, %206[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %208 = llvm.insertvalue %52, %207[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %209 = llvm.insertvalue %4, %208[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%21 : i64)
  ^bb22(%210: i64):  // 2 preds: ^bb21, ^bb23
    %211 = llvm.icmp "slt" %210, %52 : i64
    llvm.cond_br %211, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %212 = llvm.getelementptr %203[%210] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %21, %212 : !llvm.ptr<i64>
    %213 = llvm.add %210, %20  : i64
    llvm.br ^bb22(%213 : i64)
  ^bb24:  // pred: ^bb22
    %214 = llvm.alloca %4 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %209, %214 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %215 = llvm.bitcast %214 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %216 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %217 = llvm.insertvalue %4, %216[0] : !llvm.struct<(i64, ptr<i8>)> 
    %218 = llvm.insertvalue %215, %217[1] : !llvm.struct<(i64, ptr<i8>)> 
    %219 = llvm.mlir.null : !llvm.ptr<f64>
    %220 = llvm.getelementptr %219[%54] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %221 = llvm.ptrtoint %220 : !llvm.ptr<f64> to i64
    %222 = llvm.call @malloc(%221) : (i64) -> !llvm.ptr<i8>
    %223 = llvm.bitcast %222 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %224 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %225 = llvm.insertvalue %223, %224[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %226 = llvm.insertvalue %223, %225[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %227 = llvm.insertvalue %3, %226[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %228 = llvm.insertvalue %54, %227[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %229 = llvm.insertvalue %4, %228[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%21 : i64)
  ^bb25(%230: i64):  // 2 preds: ^bb24, ^bb26
    %231 = llvm.icmp "slt" %230, %54 : i64
    llvm.cond_br %231, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %232 = llvm.getelementptr %223[%230] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %17, %232 : !llvm.ptr<f64>
    %233 = llvm.add %230, %20  : i64
    llvm.br ^bb25(%233 : i64)
  ^bb27:  // pred: ^bb25
    %234 = llvm.alloca %4 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %229, %234 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %235 = llvm.bitcast %234 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %236 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %237 = llvm.insertvalue %4, %236[0] : !llvm.struct<(i64, ptr<i8>)> 
    %238 = llvm.insertvalue %235, %237[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%9, %21, %6, %20, %6, %4, %75, %4, %95, %4, %115, %4, %135, %4, %155, %4, %175, %4, %195, %4, %215, %4, %235, %10) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    %239 = llvm.mul %58, %2  : i64
    %240 = llvm.mlir.null : !llvm.ptr<f64>
    %241 = llvm.getelementptr %240[%239] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %242 = llvm.ptrtoint %241 : !llvm.ptr<f64> to i64
    %243 = llvm.add %242, %1  : i64
    %244 = llvm.call @malloc(%243) : (i64) -> !llvm.ptr<i8>
    %245 = llvm.bitcast %244 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %246 = llvm.ptrtoint %245 : !llvm.ptr<f64> to i64
    %247 = llvm.sub %1, %4  : i64
    %248 = llvm.add %246, %247  : i64
    %249 = llvm.urem %248, %1  : i64
    %250 = llvm.sub %248, %249  : i64
    %251 = llvm.inttoptr %250 : i64 to !llvm.ptr<f64>
    %252 = llvm.mul %56, %2  : i64
    %253 = llvm.mlir.null : !llvm.ptr<f64>
    %254 = llvm.getelementptr %253[%252] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %255 = llvm.ptrtoint %254 : !llvm.ptr<f64> to i64
    %256 = llvm.add %255, %1  : i64
    %257 = llvm.call @malloc(%256) : (i64) -> !llvm.ptr<i8>
    %258 = llvm.bitcast %257 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %259 = llvm.ptrtoint %258 : !llvm.ptr<f64> to i64
    %260 = llvm.sub %1, %4  : i64
    %261 = llvm.add %259, %260  : i64
    %262 = llvm.urem %261, %1  : i64
    %263 = llvm.sub %261, %262  : i64
    %264 = llvm.inttoptr %263 : i64 to !llvm.ptr<f64>
    %265 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %266 = llvm.insertvalue %258, %265[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %267 = llvm.insertvalue %264, %266[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %268 = llvm.insertvalue %3, %267[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %269 = llvm.insertvalue %56, %268[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %270 = llvm.insertvalue %2, %269[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %271 = llvm.insertvalue %2, %270[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %272 = llvm.insertvalue %4, %271[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb28(%21 : i64)
  ^bb28(%273: i64):  // 2 preds: ^bb27, ^bb31
    %274 = llvm.icmp "slt" %273, %58 : i64
    llvm.cond_br %274, ^bb29(%21 : i64), ^bb32(%21 : i64)
  ^bb29(%275: i64):  // 2 preds: ^bb28, ^bb30
    %276 = llvm.icmp "slt" %275, %19 : i64
    llvm.cond_br %276, ^bb30, ^bb31
  ^bb30:  // pred: ^bb29
    %277 = llvm.mul %273, %2  : i64
    %278 = llvm.add %277, %275  : i64
    %279 = llvm.getelementptr %251[%278] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %18, %279 : !llvm.ptr<f64>
    %280 = llvm.add %275, %20  : i64
    llvm.br ^bb29(%280 : i64)
  ^bb31:  // pred: ^bb29
    %281 = llvm.add %273, %20  : i64
    llvm.br ^bb28(%281 : i64)
  ^bb32(%282: i64):  // 2 preds: ^bb28, ^bb35
    %283 = llvm.icmp "slt" %282, %56 : i64
    llvm.cond_br %283, ^bb33(%21 : i64), ^bb36
  ^bb33(%284: i64):  // 2 preds: ^bb32, ^bb34
    %285 = llvm.icmp "slt" %284, %19 : i64
    llvm.cond_br %285, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %286 = llvm.mul %282, %2  : i64
    %287 = llvm.add %286, %284  : i64
    %288 = llvm.getelementptr %264[%287] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %17, %288 : !llvm.ptr<f64>
    %289 = llvm.add %284, %20  : i64
    llvm.br ^bb33(%289 : i64)
  ^bb35:  // pred: ^bb33
    %290 = llvm.add %282, %20  : i64
    llvm.br ^bb32(%290 : i64)
  ^bb36:  // pred: ^bb32
    %291 = llvm.load %63 : !llvm.ptr<i64>
    llvm.br ^bb37(%21 : i64)
  ^bb37(%292: i64):  // 2 preds: ^bb36, ^bb44
    %293 = llvm.icmp "slt" %292, %291 : i64
    llvm.cond_br %293, ^bb38, ^bb45
  ^bb38:  // pred: ^bb37
    %294 = llvm.add %292, %20  : i64
    %295 = llvm.getelementptr %143[%292] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %296 = llvm.load %295 : !llvm.ptr<i64>
    %297 = llvm.getelementptr %143[%294] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %298 = llvm.load %297 : !llvm.ptr<i64>
    llvm.br ^bb39(%296 : i64)
  ^bb39(%299: i64):  // 2 preds: ^bb38, ^bb43
    %300 = llvm.icmp "slt" %299, %298 : i64
    llvm.cond_br %300, ^bb40, ^bb44
  ^bb40:  // pred: ^bb39
    %301 = llvm.getelementptr %163[%299] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %302 = llvm.load %301 : !llvm.ptr<i64>
    llvm.br ^bb41(%21 : i64)
  ^bb41(%303: i64):  // 2 preds: ^bb40, ^bb42
    %304 = llvm.icmp "slt" %303, %19 : i64
    llvm.cond_br %304, ^bb42, ^bb43
  ^bb42:  // pred: ^bb41
    %305 = llvm.getelementptr %223[%299] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %306 = llvm.load %305 : !llvm.ptr<f64>
    %307 = llvm.mul %302, %2  : i64
    %308 = llvm.add %307, %303  : i64
    %309 = llvm.getelementptr %251[%308] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %310 = llvm.load %309 : !llvm.ptr<f64>
    %311 = llvm.mul %292, %2  : i64
    %312 = llvm.add %311, %303  : i64
    %313 = llvm.getelementptr %264[%312] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %314 = llvm.load %313 : !llvm.ptr<f64>
    %315 = llvm.fmul %306, %310  : f64
    %316 = llvm.fadd %314, %315  : f64
    %317 = llvm.mul %292, %2  : i64
    %318 = llvm.add %317, %303  : i64
    %319 = llvm.getelementptr %264[%318] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %316, %319 : !llvm.ptr<f64>
    %320 = llvm.add %303, %20  : i64
    llvm.br ^bb41(%320 : i64)
  ^bb43:  // pred: ^bb41
    %321 = llvm.add %299, %20  : i64
    llvm.br ^bb39(%321 : i64)
  ^bb44:  // pred: ^bb39
    llvm.br ^bb37(%294 : i64)
  ^bb45:  // pred: ^bb37
    %322 = llvm.alloca %4 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.store %272, %322 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
    %323 = llvm.bitcast %322 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
    %324 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %325 = llvm.insertvalue %0, %324[0] : !llvm.struct<(i64, ptr<i8>)> 
    %326 = llvm.insertvalue %323, %325[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @comet_print_memref_f64(%0, %323) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @comet_sort_index(i64, !llvm.ptr<i8>, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
