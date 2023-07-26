module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(12 : index) : i64
    %2 = llvm.mlir.constant(11 : index) : i64
    %3 = llvm.mlir.constant(10 : index) : i64
    %4 = llvm.mlir.constant(9 : index) : i64
    %5 = llvm.mlir.constant(8 : index) : i64
    %6 = llvm.mlir.constant(7 : index) : i64
    %7 = llvm.mlir.constant(6 : index) : i64
    %8 = llvm.mlir.constant(5 : index) : i64
    %9 = llvm.mlir.constant(4 : index) : i64
    %10 = llvm.mlir.constant(3 : index) : i64
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(-1 : index) : i64
    %17 = llvm.mlir.constant(19 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.null : !llvm.ptr<i64>
    %20 = llvm.getelementptr %19[19] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %21 = llvm.ptrtoint %20 : !llvm.ptr<i64> to i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr<i8>
    %23 = llvm.bitcast %22 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %24 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %23, %24[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.insertvalue %27, %26[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %17, %28[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %18, %29[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.alloca %31 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %30, %32 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %33 = llvm.bitcast %32 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %34 = llvm.mlir.constant(1 : index) : i64
    %35 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(i64, ptr<i8>)> 
    %37 = llvm.insertvalue %33, %36[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_3D_f64(%14, %15, %16, %15, %16, %15, %16, %34, %33, %13) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %38 = llvm.getelementptr %23[%12] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %39 = llvm.load %38 : !llvm.ptr<i64>
    %40 = llvm.getelementptr %23[%15] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %41 = llvm.load %40 : !llvm.ptr<i64>
    %42 = llvm.getelementptr %23[%11] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %43 = llvm.load %42 : !llvm.ptr<i64>
    %44 = llvm.getelementptr %23[%10] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %45 = llvm.load %44 : !llvm.ptr<i64>
    %46 = llvm.getelementptr %23[%9] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %47 = llvm.load %46 : !llvm.ptr<i64>
    %48 = llvm.getelementptr %23[%8] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %49 = llvm.load %48 : !llvm.ptr<i64>
    %50 = llvm.getelementptr %23[%7] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %51 = llvm.load %50 : !llvm.ptr<i64>
    %52 = llvm.getelementptr %23[%6] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %53 = llvm.load %52 : !llvm.ptr<i64>
    %54 = llvm.getelementptr %23[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %55 = llvm.load %54 : !llvm.ptr<i64>
    %56 = llvm.getelementptr %23[%4] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %57 = llvm.load %56 : !llvm.ptr<i64>
    %58 = llvm.getelementptr %23[%3] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %59 = llvm.load %58 : !llvm.ptr<i64>
    %60 = llvm.getelementptr %23[%2] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %61 = llvm.load %60 : !llvm.ptr<i64>
    %62 = llvm.getelementptr %23[%1] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %63 = llvm.load %62 : !llvm.ptr<i64>
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.mlir.null : !llvm.ptr<i64>
    %66 = llvm.getelementptr %65[%39] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %67 = llvm.ptrtoint %66 : !llvm.ptr<i64> to i64
    %68 = llvm.call @malloc(%67) : (i64) -> !llvm.ptr<i8>
    %69 = llvm.bitcast %68 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %70 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %71 = llvm.insertvalue %69, %70[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %73 = llvm.mlir.constant(0 : index) : i64
    %74 = llvm.insertvalue %73, %72[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.insertvalue %39, %74[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.insertvalue %64, %75[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%12 : i64)
  ^bb1(%77: i64):  // 2 preds: ^bb0, ^bb2
    %78 = llvm.icmp "slt" %77, %39 : i64
    llvm.cond_br %78, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %79 = llvm.getelementptr %69[%77] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %79 : !llvm.ptr<i64>
    %80 = llvm.add %77, %15  : i64
    llvm.br ^bb1(%80 : i64)
  ^bb3:  // pred: ^bb1
    %81 = llvm.mlir.constant(1 : index) : i64
    %82 = llvm.alloca %81 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %76, %82 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %83 = llvm.bitcast %82 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %84 = llvm.mlir.constant(1 : index) : i64
    %85 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %86 = llvm.insertvalue %84, %85[0] : !llvm.struct<(i64, ptr<i8>)> 
    %87 = llvm.insertvalue %83, %86[1] : !llvm.struct<(i64, ptr<i8>)> 
    %88 = llvm.mlir.constant(1 : index) : i64
    %89 = llvm.mlir.null : !llvm.ptr<i64>
    %90 = llvm.getelementptr %89[%41] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %91 = llvm.ptrtoint %90 : !llvm.ptr<i64> to i64
    %92 = llvm.call @malloc(%91) : (i64) -> !llvm.ptr<i8>
    %93 = llvm.bitcast %92 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %94 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %95 = llvm.insertvalue %93, %94[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.insertvalue %93, %95[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %97 = llvm.mlir.constant(0 : index) : i64
    %98 = llvm.insertvalue %97, %96[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.insertvalue %41, %98[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %100 = llvm.insertvalue %88, %99[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%12 : i64)
  ^bb4(%101: i64):  // 2 preds: ^bb3, ^bb5
    %102 = llvm.icmp "slt" %101, %41 : i64
    llvm.cond_br %102, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %103 = llvm.getelementptr %93[%101] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %103 : !llvm.ptr<i64>
    %104 = llvm.add %101, %15  : i64
    llvm.br ^bb4(%104 : i64)
  ^bb6:  // pred: ^bb4
    %105 = llvm.mlir.constant(1 : index) : i64
    %106 = llvm.alloca %105 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %100, %106 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %107 = llvm.bitcast %106 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %108 = llvm.mlir.constant(1 : index) : i64
    %109 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %110 = llvm.insertvalue %108, %109[0] : !llvm.struct<(i64, ptr<i8>)> 
    %111 = llvm.insertvalue %107, %110[1] : !llvm.struct<(i64, ptr<i8>)> 
    %112 = llvm.mlir.constant(1 : index) : i64
    %113 = llvm.mlir.null : !llvm.ptr<i64>
    %114 = llvm.getelementptr %113[%43] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %115 = llvm.ptrtoint %114 : !llvm.ptr<i64> to i64
    %116 = llvm.call @malloc(%115) : (i64) -> !llvm.ptr<i8>
    %117 = llvm.bitcast %116 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %118 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %119 = llvm.insertvalue %117, %118[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %120 = llvm.insertvalue %117, %119[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %121 = llvm.mlir.constant(0 : index) : i64
    %122 = llvm.insertvalue %121, %120[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.insertvalue %43, %122[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.insertvalue %112, %123[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%12 : i64)
  ^bb7(%125: i64):  // 2 preds: ^bb6, ^bb8
    %126 = llvm.icmp "slt" %125, %43 : i64
    llvm.cond_br %126, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %127 = llvm.getelementptr %117[%125] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %127 : !llvm.ptr<i64>
    %128 = llvm.add %125, %15  : i64
    llvm.br ^bb7(%128 : i64)
  ^bb9:  // pred: ^bb7
    %129 = llvm.mlir.constant(1 : index) : i64
    %130 = llvm.alloca %129 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %124, %130 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %131 = llvm.bitcast %130 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %132 = llvm.mlir.constant(1 : index) : i64
    %133 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %134 = llvm.insertvalue %132, %133[0] : !llvm.struct<(i64, ptr<i8>)> 
    %135 = llvm.insertvalue %131, %134[1] : !llvm.struct<(i64, ptr<i8>)> 
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.mlir.null : !llvm.ptr<i64>
    %138 = llvm.getelementptr %137[%45] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %139 = llvm.ptrtoint %138 : !llvm.ptr<i64> to i64
    %140 = llvm.call @malloc(%139) : (i64) -> !llvm.ptr<i8>
    %141 = llvm.bitcast %140 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %142 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %143 = llvm.insertvalue %141, %142[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.insertvalue %141, %143[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.mlir.constant(0 : index) : i64
    %146 = llvm.insertvalue %145, %144[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.insertvalue %45, %146[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %148 = llvm.insertvalue %136, %147[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%12 : i64)
  ^bb10(%149: i64):  // 2 preds: ^bb9, ^bb11
    %150 = llvm.icmp "slt" %149, %45 : i64
    llvm.cond_br %150, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %151 = llvm.getelementptr %141[%149] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %151 : !llvm.ptr<i64>
    %152 = llvm.add %149, %15  : i64
    llvm.br ^bb10(%152 : i64)
  ^bb12:  // pred: ^bb10
    %153 = llvm.mlir.constant(1 : index) : i64
    %154 = llvm.alloca %153 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %148, %154 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %155 = llvm.bitcast %154 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %156 = llvm.mlir.constant(1 : index) : i64
    %157 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %158 = llvm.insertvalue %156, %157[0] : !llvm.struct<(i64, ptr<i8>)> 
    %159 = llvm.insertvalue %155, %158[1] : !llvm.struct<(i64, ptr<i8>)> 
    %160 = llvm.mlir.constant(1 : index) : i64
    %161 = llvm.mlir.null : !llvm.ptr<i64>
    %162 = llvm.getelementptr %161[%47] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %163 = llvm.ptrtoint %162 : !llvm.ptr<i64> to i64
    %164 = llvm.call @malloc(%163) : (i64) -> !llvm.ptr<i8>
    %165 = llvm.bitcast %164 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %166 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %167 = llvm.insertvalue %165, %166[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %168 = llvm.insertvalue %165, %167[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %169 = llvm.mlir.constant(0 : index) : i64
    %170 = llvm.insertvalue %169, %168[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %171 = llvm.insertvalue %47, %170[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %172 = llvm.insertvalue %160, %171[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%12 : i64)
  ^bb13(%173: i64):  // 2 preds: ^bb12, ^bb14
    %174 = llvm.icmp "slt" %173, %47 : i64
    llvm.cond_br %174, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %175 = llvm.getelementptr %165[%173] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %175 : !llvm.ptr<i64>
    %176 = llvm.add %173, %15  : i64
    llvm.br ^bb13(%176 : i64)
  ^bb15:  // pred: ^bb13
    %177 = llvm.mlir.constant(1 : index) : i64
    %178 = llvm.alloca %177 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %172, %178 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %179 = llvm.bitcast %178 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %180 = llvm.mlir.constant(1 : index) : i64
    %181 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %182 = llvm.insertvalue %180, %181[0] : !llvm.struct<(i64, ptr<i8>)> 
    %183 = llvm.insertvalue %179, %182[1] : !llvm.struct<(i64, ptr<i8>)> 
    %184 = llvm.mlir.constant(1 : index) : i64
    %185 = llvm.mlir.null : !llvm.ptr<i64>
    %186 = llvm.getelementptr %185[%49] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %187 = llvm.ptrtoint %186 : !llvm.ptr<i64> to i64
    %188 = llvm.call @malloc(%187) : (i64) -> !llvm.ptr<i8>
    %189 = llvm.bitcast %188 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %190 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %191 = llvm.insertvalue %189, %190[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %192 = llvm.insertvalue %189, %191[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %193 = llvm.mlir.constant(0 : index) : i64
    %194 = llvm.insertvalue %193, %192[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %195 = llvm.insertvalue %49, %194[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %196 = llvm.insertvalue %184, %195[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%12 : i64)
  ^bb16(%197: i64):  // 2 preds: ^bb15, ^bb17
    %198 = llvm.icmp "slt" %197, %49 : i64
    llvm.cond_br %198, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %199 = llvm.getelementptr %189[%197] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %199 : !llvm.ptr<i64>
    %200 = llvm.add %197, %15  : i64
    llvm.br ^bb16(%200 : i64)
  ^bb18:  // pred: ^bb16
    %201 = llvm.mlir.constant(1 : index) : i64
    %202 = llvm.alloca %201 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %196, %202 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %203 = llvm.bitcast %202 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %204 = llvm.mlir.constant(1 : index) : i64
    %205 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %206 = llvm.insertvalue %204, %205[0] : !llvm.struct<(i64, ptr<i8>)> 
    %207 = llvm.insertvalue %203, %206[1] : !llvm.struct<(i64, ptr<i8>)> 
    %208 = llvm.mlir.constant(1 : index) : i64
    %209 = llvm.mlir.null : !llvm.ptr<i64>
    %210 = llvm.getelementptr %209[%51] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %211 = llvm.ptrtoint %210 : !llvm.ptr<i64> to i64
    %212 = llvm.call @malloc(%211) : (i64) -> !llvm.ptr<i8>
    %213 = llvm.bitcast %212 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %214 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %215 = llvm.insertvalue %213, %214[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %216 = llvm.insertvalue %213, %215[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %217 = llvm.mlir.constant(0 : index) : i64
    %218 = llvm.insertvalue %217, %216[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %219 = llvm.insertvalue %51, %218[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %220 = llvm.insertvalue %208, %219[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%12 : i64)
  ^bb19(%221: i64):  // 2 preds: ^bb18, ^bb20
    %222 = llvm.icmp "slt" %221, %51 : i64
    llvm.cond_br %222, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %223 = llvm.getelementptr %213[%221] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %223 : !llvm.ptr<i64>
    %224 = llvm.add %221, %15  : i64
    llvm.br ^bb19(%224 : i64)
  ^bb21:  // pred: ^bb19
    %225 = llvm.mlir.constant(1 : index) : i64
    %226 = llvm.alloca %225 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %220, %226 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %227 = llvm.bitcast %226 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %228 = llvm.mlir.constant(1 : index) : i64
    %229 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %230 = llvm.insertvalue %228, %229[0] : !llvm.struct<(i64, ptr<i8>)> 
    %231 = llvm.insertvalue %227, %230[1] : !llvm.struct<(i64, ptr<i8>)> 
    %232 = llvm.mlir.constant(1 : index) : i64
    %233 = llvm.mlir.null : !llvm.ptr<i64>
    %234 = llvm.getelementptr %233[%53] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %235 = llvm.ptrtoint %234 : !llvm.ptr<i64> to i64
    %236 = llvm.call @malloc(%235) : (i64) -> !llvm.ptr<i8>
    %237 = llvm.bitcast %236 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %238 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %239 = llvm.insertvalue %237, %238[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %240 = llvm.insertvalue %237, %239[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %241 = llvm.mlir.constant(0 : index) : i64
    %242 = llvm.insertvalue %241, %240[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %243 = llvm.insertvalue %53, %242[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %244 = llvm.insertvalue %232, %243[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%12 : i64)
  ^bb22(%245: i64):  // 2 preds: ^bb21, ^bb23
    %246 = llvm.icmp "slt" %245, %53 : i64
    llvm.cond_br %246, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %247 = llvm.getelementptr %237[%245] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %247 : !llvm.ptr<i64>
    %248 = llvm.add %245, %15  : i64
    llvm.br ^bb22(%248 : i64)
  ^bb24:  // pred: ^bb22
    %249 = llvm.mlir.constant(1 : index) : i64
    %250 = llvm.alloca %249 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %244, %250 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %251 = llvm.bitcast %250 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %252 = llvm.mlir.constant(1 : index) : i64
    %253 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %254 = llvm.insertvalue %252, %253[0] : !llvm.struct<(i64, ptr<i8>)> 
    %255 = llvm.insertvalue %251, %254[1] : !llvm.struct<(i64, ptr<i8>)> 
    %256 = llvm.mlir.constant(1 : index) : i64
    %257 = llvm.mlir.null : !llvm.ptr<i64>
    %258 = llvm.getelementptr %257[%55] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %259 = llvm.ptrtoint %258 : !llvm.ptr<i64> to i64
    %260 = llvm.call @malloc(%259) : (i64) -> !llvm.ptr<i8>
    %261 = llvm.bitcast %260 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %262 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %263 = llvm.insertvalue %261, %262[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %264 = llvm.insertvalue %261, %263[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %265 = llvm.mlir.constant(0 : index) : i64
    %266 = llvm.insertvalue %265, %264[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %267 = llvm.insertvalue %55, %266[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %268 = llvm.insertvalue %256, %267[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%12 : i64)
  ^bb25(%269: i64):  // 2 preds: ^bb24, ^bb26
    %270 = llvm.icmp "slt" %269, %55 : i64
    llvm.cond_br %270, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %271 = llvm.getelementptr %261[%269] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %271 : !llvm.ptr<i64>
    %272 = llvm.add %269, %15  : i64
    llvm.br ^bb25(%272 : i64)
  ^bb27:  // pred: ^bb25
    %273 = llvm.mlir.constant(1 : index) : i64
    %274 = llvm.alloca %273 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %268, %274 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %275 = llvm.bitcast %274 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %276 = llvm.mlir.constant(1 : index) : i64
    %277 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %278 = llvm.insertvalue %276, %277[0] : !llvm.struct<(i64, ptr<i8>)> 
    %279 = llvm.insertvalue %275, %278[1] : !llvm.struct<(i64, ptr<i8>)> 
    %280 = llvm.mlir.constant(1 : index) : i64
    %281 = llvm.mlir.null : !llvm.ptr<i64>
    %282 = llvm.getelementptr %281[%57] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %283 = llvm.ptrtoint %282 : !llvm.ptr<i64> to i64
    %284 = llvm.call @malloc(%283) : (i64) -> !llvm.ptr<i8>
    %285 = llvm.bitcast %284 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %286 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %287 = llvm.insertvalue %285, %286[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %288 = llvm.insertvalue %285, %287[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %289 = llvm.mlir.constant(0 : index) : i64
    %290 = llvm.insertvalue %289, %288[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %291 = llvm.insertvalue %57, %290[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %292 = llvm.insertvalue %280, %291[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb28(%12 : i64)
  ^bb28(%293: i64):  // 2 preds: ^bb27, ^bb29
    %294 = llvm.icmp "slt" %293, %57 : i64
    llvm.cond_br %294, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    %295 = llvm.getelementptr %285[%293] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %295 : !llvm.ptr<i64>
    %296 = llvm.add %293, %15  : i64
    llvm.br ^bb28(%296 : i64)
  ^bb30:  // pred: ^bb28
    %297 = llvm.mlir.constant(1 : index) : i64
    %298 = llvm.alloca %297 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %292, %298 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %299 = llvm.bitcast %298 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %300 = llvm.mlir.constant(1 : index) : i64
    %301 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %302 = llvm.insertvalue %300, %301[0] : !llvm.struct<(i64, ptr<i8>)> 
    %303 = llvm.insertvalue %299, %302[1] : !llvm.struct<(i64, ptr<i8>)> 
    %304 = llvm.mlir.constant(1 : index) : i64
    %305 = llvm.mlir.null : !llvm.ptr<i64>
    %306 = llvm.getelementptr %305[%59] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %307 = llvm.ptrtoint %306 : !llvm.ptr<i64> to i64
    %308 = llvm.call @malloc(%307) : (i64) -> !llvm.ptr<i8>
    %309 = llvm.bitcast %308 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %310 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %311 = llvm.insertvalue %309, %310[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %312 = llvm.insertvalue %309, %311[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %313 = llvm.mlir.constant(0 : index) : i64
    %314 = llvm.insertvalue %313, %312[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %315 = llvm.insertvalue %59, %314[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %316 = llvm.insertvalue %304, %315[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb31(%12 : i64)
  ^bb31(%317: i64):  // 2 preds: ^bb30, ^bb32
    %318 = llvm.icmp "slt" %317, %59 : i64
    llvm.cond_br %318, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %319 = llvm.getelementptr %309[%317] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %319 : !llvm.ptr<i64>
    %320 = llvm.add %317, %15  : i64
    llvm.br ^bb31(%320 : i64)
  ^bb33:  // pred: ^bb31
    %321 = llvm.mlir.constant(1 : index) : i64
    %322 = llvm.alloca %321 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %316, %322 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %323 = llvm.bitcast %322 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %324 = llvm.mlir.constant(1 : index) : i64
    %325 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %326 = llvm.insertvalue %324, %325[0] : !llvm.struct<(i64, ptr<i8>)> 
    %327 = llvm.insertvalue %323, %326[1] : !llvm.struct<(i64, ptr<i8>)> 
    %328 = llvm.mlir.constant(1 : index) : i64
    %329 = llvm.mlir.null : !llvm.ptr<i64>
    %330 = llvm.getelementptr %329[%61] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %331 = llvm.ptrtoint %330 : !llvm.ptr<i64> to i64
    %332 = llvm.call @malloc(%331) : (i64) -> !llvm.ptr<i8>
    %333 = llvm.bitcast %332 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %334 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %335 = llvm.insertvalue %333, %334[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %336 = llvm.insertvalue %333, %335[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %337 = llvm.mlir.constant(0 : index) : i64
    %338 = llvm.insertvalue %337, %336[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %339 = llvm.insertvalue %61, %338[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %340 = llvm.insertvalue %328, %339[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb34(%12 : i64)
  ^bb34(%341: i64):  // 2 preds: ^bb33, ^bb35
    %342 = llvm.icmp "slt" %341, %61 : i64
    llvm.cond_br %342, ^bb35, ^bb36
  ^bb35:  // pred: ^bb34
    %343 = llvm.getelementptr %333[%341] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %343 : !llvm.ptr<i64>
    %344 = llvm.add %341, %15  : i64
    llvm.br ^bb34(%344 : i64)
  ^bb36:  // pred: ^bb34
    %345 = llvm.mlir.constant(1 : index) : i64
    %346 = llvm.alloca %345 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %340, %346 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %347 = llvm.bitcast %346 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %348 = llvm.mlir.constant(1 : index) : i64
    %349 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %350 = llvm.insertvalue %348, %349[0] : !llvm.struct<(i64, ptr<i8>)> 
    %351 = llvm.insertvalue %347, %350[1] : !llvm.struct<(i64, ptr<i8>)> 
    %352 = llvm.mlir.constant(1 : index) : i64
    %353 = llvm.mlir.null : !llvm.ptr<f64>
    %354 = llvm.getelementptr %353[%63] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %355 = llvm.ptrtoint %354 : !llvm.ptr<f64> to i64
    %356 = llvm.call @malloc(%355) : (i64) -> !llvm.ptr<i8>
    %357 = llvm.bitcast %356 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %358 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %359 = llvm.insertvalue %357, %358[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %360 = llvm.insertvalue %357, %359[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %361 = llvm.mlir.constant(0 : index) : i64
    %362 = llvm.insertvalue %361, %360[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %363 = llvm.insertvalue %63, %362[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %364 = llvm.insertvalue %352, %363[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb37(%12 : i64)
  ^bb37(%365: i64):  // 2 preds: ^bb36, ^bb38
    %366 = llvm.icmp "slt" %365, %63 : i64
    llvm.cond_br %366, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %367 = llvm.getelementptr %357[%365] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %367 : !llvm.ptr<f64>
    %368 = llvm.add %365, %15  : i64
    llvm.br ^bb37(%368 : i64)
  ^bb39:  // pred: ^bb37
    %369 = llvm.mlir.constant(1 : index) : i64
    %370 = llvm.alloca %369 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %364, %370 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %371 = llvm.bitcast %370 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %372 = llvm.mlir.constant(1 : index) : i64
    %373 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %374 = llvm.insertvalue %372, %373[0] : !llvm.struct<(i64, ptr<i8>)> 
    %375 = llvm.insertvalue %371, %374[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_3D_f64(%14, %15, %16, %15, %16, %15, %16, %84, %83, %108, %107, %132, %131, %156, %155, %180, %179, %204, %203, %228, %227, %252, %251, %276, %275, %300, %299, %324, %323, %348, %347, %372, %371, %13) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    %376 = llvm.mlir.constant(1 : index) : i64
    %377 = llvm.mlir.constant(1 : index) : i64
    %378 = llvm.mlir.null : !llvm.ptr<f64>
    %379 = llvm.getelementptr %378[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
    %380 = llvm.ptrtoint %379 : !llvm.ptr<f64> to i64
    %381 = llvm.call @malloc(%380) : (i64) -> !llvm.ptr<i8>
    %382 = llvm.bitcast %381 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %383 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %384 = llvm.insertvalue %382, %383[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %385 = llvm.insertvalue %382, %384[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %386 = llvm.mlir.constant(0 : index) : i64
    %387 = llvm.insertvalue %386, %385[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %388 = llvm.insertvalue %376, %387[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %389 = llvm.insertvalue %377, %388[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %390 = llvm.getelementptr %382[%12] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %390 : !llvm.ptr<f64>
    llvm.br ^bb40(%12 : i64)
  ^bb40(%391: i64):  // 2 preds: ^bb39, ^bb41
    %392 = llvm.icmp "slt" %391, %63 : i64
    llvm.cond_br %392, ^bb41, ^bb42
  ^bb41:  // pred: ^bb40
    %393 = llvm.getelementptr %357[%391] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %394 = llvm.load %393 : !llvm.ptr<f64>
    %395 = llvm.getelementptr %382[%12] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %396 = llvm.load %395 : !llvm.ptr<f64>
    %397 = llvm.fadd %394, %396  : f64
    %398 = llvm.getelementptr %382[%12] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %397, %398 : !llvm.ptr<f64>
    %399 = llvm.add %391, %15  : i64
    llvm.br ^bb40(%399 : i64)
  ^bb42:  // pred: ^bb40
    %400 = llvm.mlir.constant(1 : index) : i64
    %401 = llvm.alloca %400 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %389, %401 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %402 = llvm.bitcast %401 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %403 = llvm.mlir.constant(1 : index) : i64
    %404 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %405 = llvm.insertvalue %403, %404[0] : !llvm.struct<(i64, ptr<i8>)> 
    %406 = llvm.insertvalue %402, %405[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @comet_print_memref_f64(%403, %402) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_3D_f64(i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_3D_f64(i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @quick_sort(i64, !llvm.ptr<i8>, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
