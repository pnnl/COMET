module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(13 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.constant(4 : index) : i64
    %10 = llvm.mlir.constant(5 : index) : i64
    %11 = llvm.mlir.constant(6 : index) : i64
    %12 = llvm.mlir.constant(7 : index) : i64
    %13 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %14 = llvm.mlir.constant(8 : index) : i64
    %15 = llvm.mlir.null : !llvm.ptr<i64>
    %16 = llvm.getelementptr %15[13] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %17 = llvm.ptrtoint %16 : !llvm.ptr<i64> to i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr<i8>
    %19 = llvm.bitcast %18 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %0, %22[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %2, %23[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %1, %24[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %25, %26 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %27 = llvm.bitcast %26 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %28 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %29 = llvm.insertvalue %1, %28[0] : !llvm.struct<(i64, ptr<i8>)> 
    %30 = llvm.insertvalue %27, %29[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_2D_f64(%7, %4, %4, %5, %4, %1, %27, %8) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %31 = llvm.load %19 : !llvm.ptr<i64>
    %32 = llvm.getelementptr %19[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %33 = llvm.load %32 : !llvm.ptr<i64>
    %34 = llvm.getelementptr %19[2] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %35 = llvm.load %34 : !llvm.ptr<i64>
    %36 = llvm.getelementptr %19[3] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %37 = llvm.load %36 : !llvm.ptr<i64>
    %38 = llvm.getelementptr %19[4] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %39 = llvm.load %38 : !llvm.ptr<i64>
    %40 = llvm.getelementptr %19[5] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %41 = llvm.load %40 : !llvm.ptr<i64>
    %42 = llvm.getelementptr %19[6] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %43 = llvm.load %42 : !llvm.ptr<i64>
    %44 = llvm.getelementptr %19[7] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %45 = llvm.load %44 : !llvm.ptr<i64>
    %46 = llvm.getelementptr %19[8] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %47 = llvm.load %46 : !llvm.ptr<i64>
    %48 = llvm.mlir.null : !llvm.ptr<i64>
    %49 = llvm.getelementptr %48[%31] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %50 = llvm.ptrtoint %49 : !llvm.ptr<i64> to i64
    %51 = llvm.call @malloc(%50) : (i64) -> !llvm.ptr<i8>
    %52 = llvm.bitcast %51 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %53 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.insertvalue %52, %53[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %0, %55[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %31, %56[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.insertvalue %1, %57[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%4 : i64)
  ^bb1(%59: i64):  // 2 preds: ^bb0, ^bb2
    %60 = llvm.icmp "slt" %59, %31 : i64
    llvm.cond_br %60, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %61 = llvm.getelementptr %52[%59] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %61 : !llvm.ptr<i64>
    %62 = llvm.add %59, %3  : i64
    llvm.br ^bb1(%62 : i64)
  ^bb3:  // pred: ^bb1
    %63 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %58, %63 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %64 = llvm.bitcast %63 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %65 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %66 = llvm.insertvalue %1, %65[0] : !llvm.struct<(i64, ptr<i8>)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(i64, ptr<i8>)> 
    %68 = llvm.mlir.null : !llvm.ptr<i64>
    %69 = llvm.getelementptr %68[%33] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %70 = llvm.ptrtoint %69 : !llvm.ptr<i64> to i64
    %71 = llvm.call @malloc(%70) : (i64) -> !llvm.ptr<i8>
    %72 = llvm.bitcast %71 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %73 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.insertvalue %72, %74[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.insertvalue %0, %75[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.insertvalue %33, %76[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.insertvalue %1, %77[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%4 : i64)
  ^bb4(%79: i64):  // 2 preds: ^bb3, ^bb5
    %80 = llvm.icmp "slt" %79, %33 : i64
    llvm.cond_br %80, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %81 = llvm.getelementptr %72[%79] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %81 : !llvm.ptr<i64>
    %82 = llvm.add %79, %3  : i64
    llvm.br ^bb4(%82 : i64)
  ^bb6:  // pred: ^bb4
    %83 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %78, %83 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %84 = llvm.bitcast %83 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %85 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %86 = llvm.insertvalue %1, %85[0] : !llvm.struct<(i64, ptr<i8>)> 
    %87 = llvm.insertvalue %84, %86[1] : !llvm.struct<(i64, ptr<i8>)> 
    %88 = llvm.mlir.null : !llvm.ptr<i64>
    %89 = llvm.getelementptr %88[%35] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %90 = llvm.ptrtoint %89 : !llvm.ptr<i64> to i64
    %91 = llvm.call @malloc(%90) : (i64) -> !llvm.ptr<i8>
    %92 = llvm.bitcast %91 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %93 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %94 = llvm.insertvalue %92, %93[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.insertvalue %0, %95[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %97 = llvm.insertvalue %35, %96[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.insertvalue %1, %97[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%4 : i64)
  ^bb7(%99: i64):  // 2 preds: ^bb6, ^bb8
    %100 = llvm.icmp "slt" %99, %35 : i64
    llvm.cond_br %100, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %101 = llvm.getelementptr %92[%99] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %101 : !llvm.ptr<i64>
    %102 = llvm.add %99, %3  : i64
    llvm.br ^bb7(%102 : i64)
  ^bb9:  // pred: ^bb7
    %103 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %98, %103 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %104 = llvm.bitcast %103 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %105 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %106 = llvm.insertvalue %1, %105[0] : !llvm.struct<(i64, ptr<i8>)> 
    %107 = llvm.insertvalue %104, %106[1] : !llvm.struct<(i64, ptr<i8>)> 
    %108 = llvm.mlir.null : !llvm.ptr<i64>
    %109 = llvm.getelementptr %108[%37] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %110 = llvm.ptrtoint %109 : !llvm.ptr<i64> to i64
    %111 = llvm.call @malloc(%110) : (i64) -> !llvm.ptr<i8>
    %112 = llvm.bitcast %111 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %113 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %114 = llvm.insertvalue %112, %113[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %115 = llvm.insertvalue %112, %114[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %116 = llvm.insertvalue %0, %115[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %117 = llvm.insertvalue %37, %116[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %118 = llvm.insertvalue %1, %117[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%4 : i64)
  ^bb10(%119: i64):  // 2 preds: ^bb9, ^bb11
    %120 = llvm.icmp "slt" %119, %37 : i64
    llvm.cond_br %120, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %121 = llvm.getelementptr %112[%119] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %121 : !llvm.ptr<i64>
    %122 = llvm.add %119, %3  : i64
    llvm.br ^bb10(%122 : i64)
  ^bb12:  // pred: ^bb10
    %123 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %118, %123 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %124 = llvm.bitcast %123 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %125 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %126 = llvm.insertvalue %1, %125[0] : !llvm.struct<(i64, ptr<i8>)> 
    %127 = llvm.insertvalue %124, %126[1] : !llvm.struct<(i64, ptr<i8>)> 
    %128 = llvm.mlir.null : !llvm.ptr<i64>
    %129 = llvm.getelementptr %128[%39] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %130 = llvm.ptrtoint %129 : !llvm.ptr<i64> to i64
    %131 = llvm.call @malloc(%130) : (i64) -> !llvm.ptr<i8>
    %132 = llvm.bitcast %131 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %133 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %134 = llvm.insertvalue %132, %133[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %135 = llvm.insertvalue %132, %134[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %136 = llvm.insertvalue %0, %135[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.insertvalue %39, %136[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %138 = llvm.insertvalue %1, %137[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%4 : i64)
  ^bb13(%139: i64):  // 2 preds: ^bb12, ^bb14
    %140 = llvm.icmp "slt" %139, %39 : i64
    llvm.cond_br %140, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %141 = llvm.getelementptr %132[%139] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %141 : !llvm.ptr<i64>
    %142 = llvm.add %139, %3  : i64
    llvm.br ^bb13(%142 : i64)
  ^bb15:  // pred: ^bb13
    %143 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %138, %143 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %144 = llvm.bitcast %143 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %145 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %146 = llvm.insertvalue %1, %145[0] : !llvm.struct<(i64, ptr<i8>)> 
    %147 = llvm.insertvalue %144, %146[1] : !llvm.struct<(i64, ptr<i8>)> 
    %148 = llvm.mlir.null : !llvm.ptr<i64>
    %149 = llvm.getelementptr %148[%41] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %150 = llvm.ptrtoint %149 : !llvm.ptr<i64> to i64
    %151 = llvm.call @malloc(%150) : (i64) -> !llvm.ptr<i8>
    %152 = llvm.bitcast %151 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %153 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %154 = llvm.insertvalue %152, %153[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %155 = llvm.insertvalue %152, %154[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %156 = llvm.insertvalue %0, %155[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %157 = llvm.insertvalue %41, %156[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %158 = llvm.insertvalue %1, %157[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%4 : i64)
  ^bb16(%159: i64):  // 2 preds: ^bb15, ^bb17
    %160 = llvm.icmp "slt" %159, %41 : i64
    llvm.cond_br %160, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %161 = llvm.getelementptr %152[%159] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %161 : !llvm.ptr<i64>
    %162 = llvm.add %159, %3  : i64
    llvm.br ^bb16(%162 : i64)
  ^bb18:  // pred: ^bb16
    %163 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %158, %163 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %164 = llvm.bitcast %163 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %165 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %166 = llvm.insertvalue %1, %165[0] : !llvm.struct<(i64, ptr<i8>)> 
    %167 = llvm.insertvalue %164, %166[1] : !llvm.struct<(i64, ptr<i8>)> 
    %168 = llvm.mlir.null : !llvm.ptr<i64>
    %169 = llvm.getelementptr %168[%43] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %170 = llvm.ptrtoint %169 : !llvm.ptr<i64> to i64
    %171 = llvm.call @malloc(%170) : (i64) -> !llvm.ptr<i8>
    %172 = llvm.bitcast %171 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %173 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %174 = llvm.insertvalue %172, %173[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %175 = llvm.insertvalue %172, %174[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %176 = llvm.insertvalue %0, %175[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %177 = llvm.insertvalue %43, %176[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %178 = llvm.insertvalue %1, %177[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%4 : i64)
  ^bb19(%179: i64):  // 2 preds: ^bb18, ^bb20
    %180 = llvm.icmp "slt" %179, %43 : i64
    llvm.cond_br %180, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %181 = llvm.getelementptr %172[%179] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %181 : !llvm.ptr<i64>
    %182 = llvm.add %179, %3  : i64
    llvm.br ^bb19(%182 : i64)
  ^bb21:  // pred: ^bb19
    %183 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %178, %183 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %184 = llvm.bitcast %183 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %185 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %186 = llvm.insertvalue %1, %185[0] : !llvm.struct<(i64, ptr<i8>)> 
    %187 = llvm.insertvalue %184, %186[1] : !llvm.struct<(i64, ptr<i8>)> 
    %188 = llvm.mlir.null : !llvm.ptr<i64>
    %189 = llvm.getelementptr %188[%45] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %190 = llvm.ptrtoint %189 : !llvm.ptr<i64> to i64
    %191 = llvm.call @malloc(%190) : (i64) -> !llvm.ptr<i8>
    %192 = llvm.bitcast %191 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %193 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %194 = llvm.insertvalue %192, %193[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %195 = llvm.insertvalue %192, %194[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %196 = llvm.insertvalue %0, %195[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %197 = llvm.insertvalue %45, %196[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %198 = llvm.insertvalue %1, %197[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%4 : i64)
  ^bb22(%199: i64):  // 2 preds: ^bb21, ^bb23
    %200 = llvm.icmp "slt" %199, %45 : i64
    llvm.cond_br %200, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %201 = llvm.getelementptr %192[%199] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %4, %201 : !llvm.ptr<i64>
    %202 = llvm.add %199, %3  : i64
    llvm.br ^bb22(%202 : i64)
  ^bb24:  // pred: ^bb22
    %203 = llvm.alloca %1 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %198, %203 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %204 = llvm.bitcast %203 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %205 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %206 = llvm.insertvalue %1, %205[0] : !llvm.struct<(i64, ptr<i8>)> 
    %207 = llvm.insertvalue %204, %206[1] : !llvm.struct<(i64, ptr<i8>)> 
    %208 = llvm.mlir.null : !llvm.ptr<f64>
    %209 = llvm.getelementptr %208[%47] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %210 = llvm.ptrtoint %209 : !llvm.ptr<f64> to i64
    %211 = llvm.call @malloc(%210) : (i64) -> !llvm.ptr<i8>
    %212 = llvm.bitcast %211 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %213 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %214 = llvm.insertvalue %212, %213[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %215 = llvm.insertvalue %212, %214[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %216 = llvm.insertvalue %0, %215[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %217 = llvm.insertvalue %47, %216[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %218 = llvm.insertvalue %1, %217[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%4 : i64)
  ^bb25(%219: i64):  // 2 preds: ^bb24, ^bb26
    %220 = llvm.icmp "slt" %219, %47 : i64
    llvm.cond_br %220, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %221 = llvm.getelementptr %212[%219] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %13, %221 : !llvm.ptr<f64>
    %222 = llvm.add %219, %3  : i64
    llvm.br ^bb25(%222 : i64)
  ^bb27:  // pred: ^bb25
    %223 = llvm.alloca %1 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %218, %223 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %224 = llvm.bitcast %223 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %225 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %226 = llvm.insertvalue %1, %225[0] : !llvm.struct<(i64, ptr<i8>)> 
    %227 = llvm.insertvalue %224, %226[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%7, %4, %4, %5, %4, %1, %64, %1, %84, %1, %104, %1, %124, %1, %144, %1, %164, %1, %184, %1, %204, %1, %224, %8) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    llvm.call @comet_print_memref_i64(%1, %64) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%1, %84) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%1, %104) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%1, %124) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%1, %144) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%1, %164) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%1, %184) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%1, %204) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_f64(%1, %224) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @comet_sort_index(i64, !llvm.ptr<i8>, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_i64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
