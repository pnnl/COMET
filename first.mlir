module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(32 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(13 : index) : i64
    %4 = llvm.mlir.constant(-1 : index) : i64
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.constant(4 : index) : i64
    %10 = llvm.mlir.constant(5 : index) : i64
    %11 = llvm.mlir.constant(6 : index) : i64
    %12 = llvm.mlir.constant(7 : index) : i64
    %13 = llvm.mlir.constant(8 : index) : i64
    %14 = llvm.mlir.constant(9 : index) : i64
    %15 = llvm.mlir.constant(10 : index) : i64
    %16 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %17 = llvm.mlir.constant(1.700000e+00 : f64) : f64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.mlir.null : !llvm.ptr<i64>
    %21 = llvm.getelementptr %20[13] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %22 = llvm.ptrtoint %21 : !llvm.ptr<i64> to i64
    %23 = llvm.call @malloc(%22) : (i64) -> !llvm.ptr<i8>
    %24 = llvm.bitcast %23 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %25 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %1, %27[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %3, %28[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %2, %29[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %30, %31 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %32 = llvm.bitcast %31 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %33 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %34 = llvm.insertvalue %2, %33[0] : !llvm.struct<(i64, ptr<i8>)> 
    %35 = llvm.insertvalue %32, %34[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_2D_f64(%7, %19, %4, %6, %19, %2, %32, %8) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %36 = llvm.load %24 : !llvm.ptr<i64>
    %37 = llvm.getelementptr %24[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %38 = llvm.load %37 : !llvm.ptr<i64>
    %39 = llvm.getelementptr %24[2] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %40 = llvm.load %39 : !llvm.ptr<i64>
    %41 = llvm.getelementptr %24[3] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %42 = llvm.load %41 : !llvm.ptr<i64>
    %43 = llvm.getelementptr %24[4] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %44 = llvm.load %43 : !llvm.ptr<i64>
    %45 = llvm.getelementptr %24[5] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %46 = llvm.load %45 : !llvm.ptr<i64>
    %47 = llvm.getelementptr %24[6] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %48 = llvm.load %47 : !llvm.ptr<i64>
    %49 = llvm.getelementptr %24[7] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %50 = llvm.load %49 : !llvm.ptr<i64>
    %51 = llvm.getelementptr %24[8] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %52 = llvm.load %51 : !llvm.ptr<i64>
    %53 = llvm.getelementptr %24[9] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %54 = llvm.load %53 : !llvm.ptr<i64>
    %55 = llvm.getelementptr %24[10] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %56 = llvm.load %55 : !llvm.ptr<i64>
    %57 = llvm.mlir.null : !llvm.ptr<i64>
    %58 = llvm.getelementptr %57[%36] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %59 = llvm.ptrtoint %58 : !llvm.ptr<i64> to i64
    %60 = llvm.call @malloc(%59) : (i64) -> !llvm.ptr<i8>
    %61 = llvm.bitcast %60 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %62 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %63 = llvm.insertvalue %61, %62[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %64 = llvm.insertvalue %61, %63[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %1, %64[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %36, %65[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %2, %66[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%19 : i64)
  ^bb1(%68: i64):  // 2 preds: ^bb0, ^bb2
    %69 = llvm.icmp "slt" %68, %36 : i64
    llvm.cond_br %69, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %70 = llvm.getelementptr %61[%68] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %70 : !llvm.ptr<i64>
    %71 = llvm.add %68, %18  : i64
    llvm.br ^bb1(%71 : i64)
  ^bb3:  // pred: ^bb1
    %72 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %67, %72 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %73 = llvm.bitcast %72 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %74 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %75 = llvm.insertvalue %2, %74[0] : !llvm.struct<(i64, ptr<i8>)> 
    %76 = llvm.insertvalue %73, %75[1] : !llvm.struct<(i64, ptr<i8>)> 
    %77 = llvm.mlir.null : !llvm.ptr<i64>
    %78 = llvm.getelementptr %77[%38] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %79 = llvm.ptrtoint %78 : !llvm.ptr<i64> to i64
    %80 = llvm.call @malloc(%79) : (i64) -> !llvm.ptr<i8>
    %81 = llvm.bitcast %80 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %82 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %83 = llvm.insertvalue %81, %82[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.insertvalue %81, %83[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.insertvalue %1, %84[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %38, %85[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.insertvalue %2, %86[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%19 : i64)
  ^bb4(%88: i64):  // 2 preds: ^bb3, ^bb5
    %89 = llvm.icmp "slt" %88, %38 : i64
    llvm.cond_br %89, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %90 = llvm.getelementptr %81[%88] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %90 : !llvm.ptr<i64>
    %91 = llvm.add %88, %18  : i64
    llvm.br ^bb4(%91 : i64)
  ^bb6:  // pred: ^bb4
    %92 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %87, %92 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %93 = llvm.bitcast %92 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %94 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %95 = llvm.insertvalue %2, %94[0] : !llvm.struct<(i64, ptr<i8>)> 
    %96 = llvm.insertvalue %93, %95[1] : !llvm.struct<(i64, ptr<i8>)> 
    %97 = llvm.mlir.null : !llvm.ptr<i64>
    %98 = llvm.getelementptr %97[%40] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %99 = llvm.ptrtoint %98 : !llvm.ptr<i64> to i64
    %100 = llvm.call @malloc(%99) : (i64) -> !llvm.ptr<i8>
    %101 = llvm.bitcast %100 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %102 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %103 = llvm.insertvalue %101, %102[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %104 = llvm.insertvalue %101, %103[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.insertvalue %1, %104[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.insertvalue %40, %105[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.insertvalue %2, %106[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%19 : i64)
  ^bb7(%108: i64):  // 2 preds: ^bb6, ^bb8
    %109 = llvm.icmp "slt" %108, %40 : i64
    llvm.cond_br %109, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %110 = llvm.getelementptr %101[%108] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %110 : !llvm.ptr<i64>
    %111 = llvm.add %108, %18  : i64
    llvm.br ^bb7(%111 : i64)
  ^bb9:  // pred: ^bb7
    %112 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %107, %112 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %113 = llvm.bitcast %112 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %114 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %115 = llvm.insertvalue %2, %114[0] : !llvm.struct<(i64, ptr<i8>)> 
    %116 = llvm.insertvalue %113, %115[1] : !llvm.struct<(i64, ptr<i8>)> 
    %117 = llvm.mlir.null : !llvm.ptr<i64>
    %118 = llvm.getelementptr %117[%42] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %119 = llvm.ptrtoint %118 : !llvm.ptr<i64> to i64
    %120 = llvm.call @malloc(%119) : (i64) -> !llvm.ptr<i8>
    %121 = llvm.bitcast %120 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %122 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %123 = llvm.insertvalue %121, %122[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.insertvalue %121, %123[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.insertvalue %1, %124[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.insertvalue %42, %125[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.insertvalue %2, %126[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%19 : i64)
  ^bb10(%128: i64):  // 2 preds: ^bb9, ^bb11
    %129 = llvm.icmp "slt" %128, %42 : i64
    llvm.cond_br %129, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %130 = llvm.getelementptr %121[%128] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %130 : !llvm.ptr<i64>
    %131 = llvm.add %128, %18  : i64
    llvm.br ^bb10(%131 : i64)
  ^bb12:  // pred: ^bb10
    %132 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %127, %132 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %133 = llvm.bitcast %132 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %134 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %135 = llvm.insertvalue %2, %134[0] : !llvm.struct<(i64, ptr<i8>)> 
    %136 = llvm.insertvalue %133, %135[1] : !llvm.struct<(i64, ptr<i8>)> 
    %137 = llvm.mlir.null : !llvm.ptr<i64>
    %138 = llvm.getelementptr %137[%44] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %139 = llvm.ptrtoint %138 : !llvm.ptr<i64> to i64
    %140 = llvm.call @malloc(%139) : (i64) -> !llvm.ptr<i8>
    %141 = llvm.bitcast %140 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %142 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %143 = llvm.insertvalue %141, %142[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.insertvalue %141, %143[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.insertvalue %1, %144[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.insertvalue %44, %145[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.insertvalue %2, %146[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%19 : i64)
  ^bb13(%148: i64):  // 2 preds: ^bb12, ^bb14
    %149 = llvm.icmp "slt" %148, %44 : i64
    llvm.cond_br %149, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %150 = llvm.getelementptr %141[%148] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %150 : !llvm.ptr<i64>
    %151 = llvm.add %148, %18  : i64
    llvm.br ^bb13(%151 : i64)
  ^bb15:  // pred: ^bb13
    %152 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %147, %152 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %153 = llvm.bitcast %152 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %154 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %155 = llvm.insertvalue %2, %154[0] : !llvm.struct<(i64, ptr<i8>)> 
    %156 = llvm.insertvalue %153, %155[1] : !llvm.struct<(i64, ptr<i8>)> 
    %157 = llvm.mlir.null : !llvm.ptr<i64>
    %158 = llvm.getelementptr %157[%46] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %159 = llvm.ptrtoint %158 : !llvm.ptr<i64> to i64
    %160 = llvm.call @malloc(%159) : (i64) -> !llvm.ptr<i8>
    %161 = llvm.bitcast %160 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %162 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %163 = llvm.insertvalue %161, %162[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %164 = llvm.insertvalue %161, %163[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %165 = llvm.insertvalue %1, %164[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.insertvalue %46, %165[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.insertvalue %2, %166[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%19 : i64)
  ^bb16(%168: i64):  // 2 preds: ^bb15, ^bb17
    %169 = llvm.icmp "slt" %168, %46 : i64
    llvm.cond_br %169, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %170 = llvm.getelementptr %161[%168] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %170 : !llvm.ptr<i64>
    %171 = llvm.add %168, %18  : i64
    llvm.br ^bb16(%171 : i64)
  ^bb18:  // pred: ^bb16
    %172 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %167, %172 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %173 = llvm.bitcast %172 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %174 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %175 = llvm.insertvalue %2, %174[0] : !llvm.struct<(i64, ptr<i8>)> 
    %176 = llvm.insertvalue %173, %175[1] : !llvm.struct<(i64, ptr<i8>)> 
    %177 = llvm.mlir.null : !llvm.ptr<i64>
    %178 = llvm.getelementptr %177[%48] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %179 = llvm.ptrtoint %178 : !llvm.ptr<i64> to i64
    %180 = llvm.call @malloc(%179) : (i64) -> !llvm.ptr<i8>
    %181 = llvm.bitcast %180 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %182 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %183 = llvm.insertvalue %181, %182[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %184 = llvm.insertvalue %181, %183[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %185 = llvm.insertvalue %1, %184[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %186 = llvm.insertvalue %48, %185[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %187 = llvm.insertvalue %2, %186[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%19 : i64)
  ^bb19(%188: i64):  // 2 preds: ^bb18, ^bb20
    %189 = llvm.icmp "slt" %188, %48 : i64
    llvm.cond_br %189, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %190 = llvm.getelementptr %181[%188] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %190 : !llvm.ptr<i64>
    %191 = llvm.add %188, %18  : i64
    llvm.br ^bb19(%191 : i64)
  ^bb21:  // pred: ^bb19
    %192 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %187, %192 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %193 = llvm.bitcast %192 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %194 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %195 = llvm.insertvalue %2, %194[0] : !llvm.struct<(i64, ptr<i8>)> 
    %196 = llvm.insertvalue %193, %195[1] : !llvm.struct<(i64, ptr<i8>)> 
    %197 = llvm.mlir.null : !llvm.ptr<i64>
    %198 = llvm.getelementptr %197[%50] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %199 = llvm.ptrtoint %198 : !llvm.ptr<i64> to i64
    %200 = llvm.call @malloc(%199) : (i64) -> !llvm.ptr<i8>
    %201 = llvm.bitcast %200 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %202 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %203 = llvm.insertvalue %201, %202[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %204 = llvm.insertvalue %201, %203[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %205 = llvm.insertvalue %1, %204[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %206 = llvm.insertvalue %50, %205[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %207 = llvm.insertvalue %2, %206[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%19 : i64)
  ^bb22(%208: i64):  // 2 preds: ^bb21, ^bb23
    %209 = llvm.icmp "slt" %208, %50 : i64
    llvm.cond_br %209, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %210 = llvm.getelementptr %201[%208] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %210 : !llvm.ptr<i64>
    %211 = llvm.add %208, %18  : i64
    llvm.br ^bb22(%211 : i64)
  ^bb24:  // pred: ^bb22
    %212 = llvm.alloca %2 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %207, %212 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %213 = llvm.bitcast %212 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %214 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %215 = llvm.insertvalue %2, %214[0] : !llvm.struct<(i64, ptr<i8>)> 
    %216 = llvm.insertvalue %213, %215[1] : !llvm.struct<(i64, ptr<i8>)> 
    %217 = llvm.mlir.null : !llvm.ptr<f64>
    %218 = llvm.getelementptr %217[%52] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %219 = llvm.ptrtoint %218 : !llvm.ptr<f64> to i64
    %220 = llvm.call @malloc(%219) : (i64) -> !llvm.ptr<i8>
    %221 = llvm.bitcast %220 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %222 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %223 = llvm.insertvalue %221, %222[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %224 = llvm.insertvalue %221, %223[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %225 = llvm.insertvalue %1, %224[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %226 = llvm.insertvalue %52, %225[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %227 = llvm.insertvalue %2, %226[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%19 : i64)
  ^bb25(%228: i64):  // 2 preds: ^bb24, ^bb26
    %229 = llvm.icmp "slt" %228, %52 : i64
    llvm.cond_br %229, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %230 = llvm.getelementptr %221[%228] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %16, %230 : !llvm.ptr<f64>
    %231 = llvm.add %228, %18  : i64
    llvm.br ^bb25(%231 : i64)
  ^bb27:  // pred: ^bb25
    %232 = llvm.alloca %2 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %227, %232 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %233 = llvm.bitcast %232 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %234 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %235 = llvm.insertvalue %2, %234[0] : !llvm.struct<(i64, ptr<i8>)> 
    %236 = llvm.insertvalue %233, %235[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%7, %19, %4, %6, %19, %2, %73, %2, %93, %2, %113, %2, %133, %2, %153, %2, %173, %2, %193, %2, %213, %2, %233, %8) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    %237 = llvm.mlir.null : !llvm.ptr<f64>
    %238 = llvm.getelementptr %237[%56] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %239 = llvm.ptrtoint %238 : !llvm.ptr<f64> to i64
    %240 = llvm.add %239, %0  : i64
    %241 = llvm.call @malloc(%240) : (i64) -> !llvm.ptr<i8>
    %242 = llvm.bitcast %241 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %243 = llvm.ptrtoint %242 : !llvm.ptr<f64> to i64
    %244 = llvm.sub %0, %2  : i64
    %245 = llvm.add %243, %244  : i64
    %246 = llvm.urem %245, %0  : i64
    %247 = llvm.sub %245, %246  : i64
    %248 = llvm.inttoptr %247 : i64 to !llvm.ptr<f64>
    %249 = llvm.mlir.null : !llvm.ptr<f64>
    %250 = llvm.getelementptr %249[%54] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %251 = llvm.ptrtoint %250 : !llvm.ptr<f64> to i64
    %252 = llvm.add %251, %0  : i64
    %253 = llvm.call @malloc(%252) : (i64) -> !llvm.ptr<i8>
    %254 = llvm.bitcast %253 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %255 = llvm.ptrtoint %254 : !llvm.ptr<f64> to i64
    %256 = llvm.sub %0, %2  : i64
    %257 = llvm.add %255, %256  : i64
    %258 = llvm.urem %257, %0  : i64
    %259 = llvm.sub %257, %258  : i64
    %260 = llvm.inttoptr %259 : i64 to !llvm.ptr<f64>
    %261 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %262 = llvm.insertvalue %254, %261[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %263 = llvm.insertvalue %260, %262[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %264 = llvm.insertvalue %1, %263[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %265 = llvm.insertvalue %54, %264[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %266 = llvm.insertvalue %2, %265[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb28(%19 : i64)
  ^bb28(%267: i64):  // 2 preds: ^bb27, ^bb29
    %268 = llvm.icmp "slt" %267, %56 : i64
    llvm.cond_br %268, ^bb29, ^bb30(%19 : i64)
  ^bb29:  // pred: ^bb28
    %269 = llvm.getelementptr %248[%267] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %17, %269 : !llvm.ptr<f64>
    %270 = llvm.add %267, %18  : i64
    llvm.br ^bb28(%270 : i64)
  ^bb30(%271: i64):  // 2 preds: ^bb28, ^bb31
    %272 = llvm.icmp "slt" %271, %54 : i64
    llvm.cond_br %272, ^bb31, ^bb32
  ^bb31:  // pred: ^bb30
    %273 = llvm.getelementptr %260[%271] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %16, %273 : !llvm.ptr<f64>
    %274 = llvm.add %271, %18  : i64
    llvm.br ^bb30(%274 : i64)
  ^bb32:  // pred: ^bb30
    %275 = llvm.load %61 : !llvm.ptr<i64>
    llvm.br ^bb33(%19 : i64)
  ^bb33(%276: i64):  // 2 preds: ^bb32, ^bb37
    %277 = llvm.icmp "slt" %276, %275 : i64
    llvm.cond_br %277, ^bb34, ^bb38
  ^bb34:  // pred: ^bb33
    %278 = llvm.load %181 : !llvm.ptr<i64>
    llvm.br ^bb35(%19 : i64)
  ^bb35(%279: i64):  // 2 preds: ^bb34, ^bb36
    %280 = llvm.icmp "slt" %279, %278 : i64
    llvm.cond_br %280, ^bb36, ^bb37
  ^bb36:  // pred: ^bb35
    %281 = llvm.load %181 : !llvm.ptr<i64>
    %282 = llvm.mul %276, %281  : i64
    %283 = llvm.add %282, %279  : i64
    %284 = llvm.getelementptr %161[%283] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %285 = llvm.load %284 : !llvm.ptr<i64>
    %286 = llvm.getelementptr %221[%283] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %287 = llvm.load %286 : !llvm.ptr<f64>
    %288 = llvm.getelementptr %248[%285] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %289 = llvm.load %288 : !llvm.ptr<f64>
    %290 = llvm.getelementptr %260[%279] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %291 = llvm.load %290 : !llvm.ptr<f64>
    %292 = llvm.fmul %287, %289  : f64
    %293 = llvm.fadd %291, %292  : f64
    %294 = llvm.getelementptr %260[%279] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %293, %294 : !llvm.ptr<f64>
    %295 = llvm.add %279, %18  : i64
    llvm.br ^bb35(%295 : i64)
  ^bb37:  // pred: ^bb35
    %296 = llvm.add %276, %18  : i64
    llvm.br ^bb33(%296 : i64)
  ^bb38:  // pred: ^bb33
    llvm.call @comet_print_memref_i64(%2, %73) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%2, %93) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%2, %113) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%2, %133) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%2, %153) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%2, %173) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%2, %193) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%2, %213) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_f64(%2, %233) : (i64, !llvm.ptr<i8>) -> ()
    %297 = llvm.alloca %2 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %266, %297 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %298 = llvm.bitcast %297 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %299 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %300 = llvm.insertvalue %2, %299[0] : !llvm.struct<(i64, ptr<i8>)> 
    %301 = llvm.insertvalue %298, %300[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @comet_print_memref_f64(%2, %298) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @comet_sort_index(i64, !llvm.ptr<i8>, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_i64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
