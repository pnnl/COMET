module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(8 : index) : i64
    %2 = llvm.mlir.constant(7 : index) : i64
    %3 = llvm.mlir.constant(6 : index) : i64
    %4 = llvm.mlir.constant(5 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(3 : index) : i64
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(-1 : index) : i64
    %13 = llvm.mlir.constant(13 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.null : !llvm.ptr<i64>
    %16 = llvm.getelementptr %15[13] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %17 = llvm.ptrtoint %16 : !llvm.ptr<i64> to i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr<i8>
    %19 = llvm.bitcast %18 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.mlir.constant(0 : index) : i64
    %24 = llvm.insertvalue %23, %22[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %13, %24[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %14, %25[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.alloca %27 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %26, %28 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %29 = llvm.bitcast %28 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(i64, ptr<i8>)> 
    %33 = llvm.insertvalue %29, %32[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_2D_f64(%9, %11, %12, %10, %12, %30, %29, %8) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %34 = llvm.getelementptr %19[%7] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %35 = llvm.load %34 : !llvm.ptr<i64>
    %36 = llvm.getelementptr %19[%6] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %37 = llvm.load %36 : !llvm.ptr<i64>
    %38 = llvm.getelementptr %19[%11] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %39 = llvm.load %38 : !llvm.ptr<i64>
    %40 = llvm.getelementptr %19[%10] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %41 = llvm.load %40 : !llvm.ptr<i64>
    %42 = llvm.getelementptr %19[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %43 = llvm.load %42 : !llvm.ptr<i64>
    %44 = llvm.getelementptr %19[%4] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %45 = llvm.load %44 : !llvm.ptr<i64>
    %46 = llvm.getelementptr %19[%3] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %47 = llvm.load %46 : !llvm.ptr<i64>
    %48 = llvm.getelementptr %19[%2] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %49 = llvm.load %48 : !llvm.ptr<i64>
    %50 = llvm.getelementptr %19[%1] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %51 = llvm.load %50 : !llvm.ptr<i64>
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.mlir.null : !llvm.ptr<i64>
    %54 = llvm.getelementptr %53[%35] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %55 = llvm.ptrtoint %54 : !llvm.ptr<i64> to i64
    %56 = llvm.call @malloc(%55) : (i64) -> !llvm.ptr<i8>
    %57 = llvm.bitcast %56 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %58 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %59 = llvm.insertvalue %57, %58[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.insertvalue %57, %59[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %61 = llvm.mlir.constant(0 : index) : i64
    %62 = llvm.insertvalue %61, %60[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %63 = llvm.insertvalue %35, %62[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %64 = llvm.insertvalue %52, %63[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%7 : i64)
  ^bb1(%65: i64):  // 2 preds: ^bb0, ^bb2
    %66 = llvm.icmp "slt" %65, %35 : i64
    llvm.cond_br %66, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %67 = llvm.getelementptr %57[%65] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %67 : !llvm.ptr<i64>
    %68 = llvm.add %65, %6  : i64
    llvm.br ^bb1(%68 : i64)
  ^bb3:  // pred: ^bb1
    %69 = llvm.mlir.constant(1 : index) : i64
    %70 = llvm.alloca %69 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %64, %70 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %71 = llvm.bitcast %70 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %72 = llvm.mlir.constant(1 : index) : i64
    %73 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.struct<(i64, ptr<i8>)> 
    %75 = llvm.insertvalue %71, %74[1] : !llvm.struct<(i64, ptr<i8>)> 
    %76 = llvm.mlir.constant(1 : index) : i64
    %77 = llvm.mlir.null : !llvm.ptr<i64>
    %78 = llvm.getelementptr %77[%37] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %79 = llvm.ptrtoint %78 : !llvm.ptr<i64> to i64
    %80 = llvm.call @malloc(%79) : (i64) -> !llvm.ptr<i8>
    %81 = llvm.bitcast %80 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %82 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %83 = llvm.insertvalue %81, %82[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.insertvalue %81, %83[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.mlir.constant(0 : index) : i64
    %86 = llvm.insertvalue %85, %84[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.insertvalue %37, %86[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.insertvalue %76, %87[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%7 : i64)
  ^bb4(%89: i64):  // 2 preds: ^bb3, ^bb5
    %90 = llvm.icmp "slt" %89, %37 : i64
    llvm.cond_br %90, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %91 = llvm.getelementptr %81[%89] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %91 : !llvm.ptr<i64>
    %92 = llvm.add %89, %6  : i64
    llvm.br ^bb4(%92 : i64)
  ^bb6:  // pred: ^bb4
    %93 = llvm.mlir.constant(1 : index) : i64
    %94 = llvm.alloca %93 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %88, %94 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %95 = llvm.bitcast %94 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %96 = llvm.mlir.constant(1 : index) : i64
    %97 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %98 = llvm.insertvalue %96, %97[0] : !llvm.struct<(i64, ptr<i8>)> 
    %99 = llvm.insertvalue %95, %98[1] : !llvm.struct<(i64, ptr<i8>)> 
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.mlir.null : !llvm.ptr<i64>
    %102 = llvm.getelementptr %101[%39] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %103 = llvm.ptrtoint %102 : !llvm.ptr<i64> to i64
    %104 = llvm.call @malloc(%103) : (i64) -> !llvm.ptr<i8>
    %105 = llvm.bitcast %104 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %106 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %107 = llvm.insertvalue %105, %106[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %108 = llvm.insertvalue %105, %107[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %109 = llvm.mlir.constant(0 : index) : i64
    %110 = llvm.insertvalue %109, %108[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %111 = llvm.insertvalue %39, %110[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %112 = llvm.insertvalue %100, %111[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%7 : i64)
  ^bb7(%113: i64):  // 2 preds: ^bb6, ^bb8
    %114 = llvm.icmp "slt" %113, %39 : i64
    llvm.cond_br %114, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %115 = llvm.getelementptr %105[%113] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %115 : !llvm.ptr<i64>
    %116 = llvm.add %113, %6  : i64
    llvm.br ^bb7(%116 : i64)
  ^bb9:  // pred: ^bb7
    %117 = llvm.mlir.constant(1 : index) : i64
    %118 = llvm.alloca %117 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %112, %118 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %119 = llvm.bitcast %118 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %120 = llvm.mlir.constant(1 : index) : i64
    %121 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %122 = llvm.insertvalue %120, %121[0] : !llvm.struct<(i64, ptr<i8>)> 
    %123 = llvm.insertvalue %119, %122[1] : !llvm.struct<(i64, ptr<i8>)> 
    %124 = llvm.mlir.constant(1 : index) : i64
    %125 = llvm.mlir.null : !llvm.ptr<i64>
    %126 = llvm.getelementptr %125[%41] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %127 = llvm.ptrtoint %126 : !llvm.ptr<i64> to i64
    %128 = llvm.call @malloc(%127) : (i64) -> !llvm.ptr<i8>
    %129 = llvm.bitcast %128 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %130 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %131 = llvm.insertvalue %129, %130[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %132 = llvm.insertvalue %129, %131[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %133 = llvm.mlir.constant(0 : index) : i64
    %134 = llvm.insertvalue %133, %132[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %135 = llvm.insertvalue %41, %134[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %136 = llvm.insertvalue %124, %135[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%7 : i64)
  ^bb10(%137: i64):  // 2 preds: ^bb9, ^bb11
    %138 = llvm.icmp "slt" %137, %41 : i64
    llvm.cond_br %138, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %139 = llvm.getelementptr %129[%137] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %139 : !llvm.ptr<i64>
    %140 = llvm.add %137, %6  : i64
    llvm.br ^bb10(%140 : i64)
  ^bb12:  // pred: ^bb10
    %141 = llvm.mlir.constant(1 : index) : i64
    %142 = llvm.alloca %141 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %136, %142 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %143 = llvm.bitcast %142 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %144 = llvm.mlir.constant(1 : index) : i64
    %145 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %146 = llvm.insertvalue %144, %145[0] : !llvm.struct<(i64, ptr<i8>)> 
    %147 = llvm.insertvalue %143, %146[1] : !llvm.struct<(i64, ptr<i8>)> 
    %148 = llvm.mlir.constant(1 : index) : i64
    %149 = llvm.mlir.null : !llvm.ptr<i64>
    %150 = llvm.getelementptr %149[%43] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %151 = llvm.ptrtoint %150 : !llvm.ptr<i64> to i64
    %152 = llvm.call @malloc(%151) : (i64) -> !llvm.ptr<i8>
    %153 = llvm.bitcast %152 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %154 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %155 = llvm.insertvalue %153, %154[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %156 = llvm.insertvalue %153, %155[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %157 = llvm.mlir.constant(0 : index) : i64
    %158 = llvm.insertvalue %157, %156[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %159 = llvm.insertvalue %43, %158[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %160 = llvm.insertvalue %148, %159[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%7 : i64)
  ^bb13(%161: i64):  // 2 preds: ^bb12, ^bb14
    %162 = llvm.icmp "slt" %161, %43 : i64
    llvm.cond_br %162, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %163 = llvm.getelementptr %153[%161] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %163 : !llvm.ptr<i64>
    %164 = llvm.add %161, %6  : i64
    llvm.br ^bb13(%164 : i64)
  ^bb15:  // pred: ^bb13
    %165 = llvm.mlir.constant(1 : index) : i64
    %166 = llvm.alloca %165 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %160, %166 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %167 = llvm.bitcast %166 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %168 = llvm.mlir.constant(1 : index) : i64
    %169 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %170 = llvm.insertvalue %168, %169[0] : !llvm.struct<(i64, ptr<i8>)> 
    %171 = llvm.insertvalue %167, %170[1] : !llvm.struct<(i64, ptr<i8>)> 
    %172 = llvm.mlir.constant(1 : index) : i64
    %173 = llvm.mlir.null : !llvm.ptr<i64>
    %174 = llvm.getelementptr %173[%45] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %175 = llvm.ptrtoint %174 : !llvm.ptr<i64> to i64
    %176 = llvm.call @malloc(%175) : (i64) -> !llvm.ptr<i8>
    %177 = llvm.bitcast %176 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %178 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %179 = llvm.insertvalue %177, %178[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %180 = llvm.insertvalue %177, %179[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %181 = llvm.mlir.constant(0 : index) : i64
    %182 = llvm.insertvalue %181, %180[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %183 = llvm.insertvalue %45, %182[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %184 = llvm.insertvalue %172, %183[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%7 : i64)
  ^bb16(%185: i64):  // 2 preds: ^bb15, ^bb17
    %186 = llvm.icmp "slt" %185, %45 : i64
    llvm.cond_br %186, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %187 = llvm.getelementptr %177[%185] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %187 : !llvm.ptr<i64>
    %188 = llvm.add %185, %6  : i64
    llvm.br ^bb16(%188 : i64)
  ^bb18:  // pred: ^bb16
    %189 = llvm.mlir.constant(1 : index) : i64
    %190 = llvm.alloca %189 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %184, %190 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %191 = llvm.bitcast %190 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %192 = llvm.mlir.constant(1 : index) : i64
    %193 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %194 = llvm.insertvalue %192, %193[0] : !llvm.struct<(i64, ptr<i8>)> 
    %195 = llvm.insertvalue %191, %194[1] : !llvm.struct<(i64, ptr<i8>)> 
    %196 = llvm.mlir.constant(1 : index) : i64
    %197 = llvm.mlir.null : !llvm.ptr<i64>
    %198 = llvm.getelementptr %197[%47] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %199 = llvm.ptrtoint %198 : !llvm.ptr<i64> to i64
    %200 = llvm.call @malloc(%199) : (i64) -> !llvm.ptr<i8>
    %201 = llvm.bitcast %200 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %202 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %203 = llvm.insertvalue %201, %202[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %204 = llvm.insertvalue %201, %203[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %205 = llvm.mlir.constant(0 : index) : i64
    %206 = llvm.insertvalue %205, %204[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %207 = llvm.insertvalue %47, %206[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %208 = llvm.insertvalue %196, %207[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%7 : i64)
  ^bb19(%209: i64):  // 2 preds: ^bb18, ^bb20
    %210 = llvm.icmp "slt" %209, %47 : i64
    llvm.cond_br %210, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %211 = llvm.getelementptr %201[%209] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %211 : !llvm.ptr<i64>
    %212 = llvm.add %209, %6  : i64
    llvm.br ^bb19(%212 : i64)
  ^bb21:  // pred: ^bb19
    %213 = llvm.mlir.constant(1 : index) : i64
    %214 = llvm.alloca %213 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %208, %214 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %215 = llvm.bitcast %214 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %216 = llvm.mlir.constant(1 : index) : i64
    %217 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %218 = llvm.insertvalue %216, %217[0] : !llvm.struct<(i64, ptr<i8>)> 
    %219 = llvm.insertvalue %215, %218[1] : !llvm.struct<(i64, ptr<i8>)> 
    %220 = llvm.mlir.constant(1 : index) : i64
    %221 = llvm.mlir.null : !llvm.ptr<i64>
    %222 = llvm.getelementptr %221[%49] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %223 = llvm.ptrtoint %222 : !llvm.ptr<i64> to i64
    %224 = llvm.call @malloc(%223) : (i64) -> !llvm.ptr<i8>
    %225 = llvm.bitcast %224 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %226 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %227 = llvm.insertvalue %225, %226[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %228 = llvm.insertvalue %225, %227[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %229 = llvm.mlir.constant(0 : index) : i64
    %230 = llvm.insertvalue %229, %228[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %231 = llvm.insertvalue %49, %230[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %232 = llvm.insertvalue %220, %231[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%7 : i64)
  ^bb22(%233: i64):  // 2 preds: ^bb21, ^bb23
    %234 = llvm.icmp "slt" %233, %49 : i64
    llvm.cond_br %234, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %235 = llvm.getelementptr %225[%233] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %7, %235 : !llvm.ptr<i64>
    %236 = llvm.add %233, %6  : i64
    llvm.br ^bb22(%236 : i64)
  ^bb24:  // pred: ^bb22
    %237 = llvm.mlir.constant(1 : index) : i64
    %238 = llvm.alloca %237 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %232, %238 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %239 = llvm.bitcast %238 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %240 = llvm.mlir.constant(1 : index) : i64
    %241 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %242 = llvm.insertvalue %240, %241[0] : !llvm.struct<(i64, ptr<i8>)> 
    %243 = llvm.insertvalue %239, %242[1] : !llvm.struct<(i64, ptr<i8>)> 
    %244 = llvm.mlir.constant(1 : index) : i64
    %245 = llvm.mlir.null : !llvm.ptr<f64>
    %246 = llvm.getelementptr %245[%51] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %247 = llvm.ptrtoint %246 : !llvm.ptr<f64> to i64
    %248 = llvm.call @malloc(%247) : (i64) -> !llvm.ptr<i8>
    %249 = llvm.bitcast %248 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %250 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %251 = llvm.insertvalue %249, %250[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %252 = llvm.insertvalue %249, %251[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %253 = llvm.mlir.constant(0 : index) : i64
    %254 = llvm.insertvalue %253, %252[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %255 = llvm.insertvalue %51, %254[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %256 = llvm.insertvalue %244, %255[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%7 : i64)
  ^bb25(%257: i64):  // 2 preds: ^bb24, ^bb26
    %258 = llvm.icmp "slt" %257, %51 : i64
    llvm.cond_br %258, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %259 = llvm.getelementptr %249[%257] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %259 : !llvm.ptr<f64>
    %260 = llvm.add %257, %6  : i64
    llvm.br ^bb25(%260 : i64)
  ^bb27:  // pred: ^bb25
    %261 = llvm.mlir.constant(1 : index) : i64
    %262 = llvm.alloca %261 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %256, %262 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %263 = llvm.bitcast %262 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %264 = llvm.mlir.constant(1 : index) : i64
    %265 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %266 = llvm.insertvalue %264, %265[0] : !llvm.struct<(i64, ptr<i8>)> 
    %267 = llvm.insertvalue %263, %266[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%9, %11, %12, %10, %12, %72, %71, %96, %95, %120, %119, %144, %143, %168, %167, %192, %191, %216, %215, %240, %239, %264, %263, %8) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    %268 = llvm.mlir.constant(1 : index) : i64
    %269 = llvm.mlir.constant(1 : index) : i64
    %270 = llvm.mlir.null : !llvm.ptr<f64>
    %271 = llvm.getelementptr %270[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
    %272 = llvm.ptrtoint %271 : !llvm.ptr<f64> to i64
    %273 = llvm.call @malloc(%272) : (i64) -> !llvm.ptr<i8>
    %274 = llvm.bitcast %273 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %275 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %276 = llvm.insertvalue %274, %275[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %277 = llvm.insertvalue %274, %276[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %278 = llvm.mlir.constant(0 : index) : i64
    %279 = llvm.insertvalue %278, %277[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %280 = llvm.insertvalue %268, %279[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %281 = llvm.insertvalue %269, %280[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %282 = llvm.getelementptr %274[%7] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %282 : !llvm.ptr<f64>
    llvm.br ^bb28(%7 : i64)
  ^bb28(%283: i64):  // 2 preds: ^bb27, ^bb29
    %284 = llvm.icmp "slt" %283, %51 : i64
    llvm.cond_br %284, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    %285 = llvm.getelementptr %249[%283] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %286 = llvm.load %285 : !llvm.ptr<f64>
    %287 = llvm.getelementptr %274[%7] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %288 = llvm.load %287 : !llvm.ptr<f64>
    %289 = llvm.fadd %286, %288  : f64
    %290 = llvm.getelementptr %274[%7] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %289, %290 : !llvm.ptr<f64>
    %291 = llvm.add %283, %6  : i64
    llvm.br ^bb28(%291 : i64)
  ^bb30:  // pred: ^bb28
    %292 = llvm.mlir.constant(1 : index) : i64
    %293 = llvm.alloca %292 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %281, %293 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %294 = llvm.bitcast %293 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %295 = llvm.mlir.constant(1 : index) : i64
    %296 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %297 = llvm.insertvalue %295, %296[0] : !llvm.struct<(i64, ptr<i8>)> 
    %298 = llvm.insertvalue %294, %297[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @comet_print_memref_f64(%295, %294) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @quick_sort(i64, !llvm.ptr<i8>, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
