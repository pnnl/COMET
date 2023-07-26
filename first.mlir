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
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(-1 : index) : i64
    %16 = llvm.mlir.constant(19 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.null : !llvm.ptr<i64>
    %19 = llvm.getelementptr %18[19] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
    %20 = llvm.ptrtoint %19 : !llvm.ptr<i64> to i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr<i8>
    %22 = llvm.bitcast %21 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %23 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %16, %27[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %17, %28[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.alloca %30 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %29, %31 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %32 = llvm.bitcast %31 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(i64, ptr<i8>)> 
    %36 = llvm.insertvalue %32, %35[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_sizes_3D_f64(%13, %14, %15, %14, %15, %14, %15, %33, %32, %13) {filename = "SPARSE_FILE_NAME1"} : (i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %37 = llvm.getelementptr %22[%12] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %38 = llvm.load %37 : !llvm.ptr<i64>
    %39 = llvm.getelementptr %22[%14] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %40 = llvm.load %39 : !llvm.ptr<i64>
    %41 = llvm.getelementptr %22[%11] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %42 = llvm.load %41 : !llvm.ptr<i64>
    %43 = llvm.getelementptr %22[%10] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %44 = llvm.load %43 : !llvm.ptr<i64>
    %45 = llvm.getelementptr %22[%9] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %46 = llvm.load %45 : !llvm.ptr<i64>
    %47 = llvm.getelementptr %22[%8] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %48 = llvm.load %47 : !llvm.ptr<i64>
    %49 = llvm.getelementptr %22[%7] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %50 = llvm.load %49 : !llvm.ptr<i64>
    %51 = llvm.getelementptr %22[%6] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %52 = llvm.load %51 : !llvm.ptr<i64>
    %53 = llvm.getelementptr %22[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %54 = llvm.load %53 : !llvm.ptr<i64>
    %55 = llvm.getelementptr %22[%4] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %56 = llvm.load %55 : !llvm.ptr<i64>
    %57 = llvm.getelementptr %22[%3] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %58 = llvm.load %57 : !llvm.ptr<i64>
    %59 = llvm.getelementptr %22[%2] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %60 = llvm.load %59 : !llvm.ptr<i64>
    %61 = llvm.getelementptr %22[%1] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %62 = llvm.load %61 : !llvm.ptr<i64>
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.mlir.null : !llvm.ptr<i64>
    %65 = llvm.getelementptr %64[%38] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %66 = llvm.ptrtoint %65 : !llvm.ptr<i64> to i64
    %67 = llvm.call @malloc(%66) : (i64) -> !llvm.ptr<i8>
    %68 = llvm.bitcast %67 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %69 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %71 = llvm.insertvalue %68, %70[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.insertvalue %38, %73[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.insertvalue %63, %74[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%12 : i64)
  ^bb1(%76: i64):  // 2 preds: ^bb0, ^bb2
    %77 = llvm.icmp "slt" %76, %38 : i64
    llvm.cond_br %77, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %78 = llvm.getelementptr %68[%76] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %78 : !llvm.ptr<i64>
    %79 = llvm.add %76, %14  : i64
    llvm.br ^bb1(%79 : i64)
  ^bb3:  // pred: ^bb1
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
    %89 = llvm.getelementptr %88[%40] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %90 = llvm.ptrtoint %89 : !llvm.ptr<i64> to i64
    %91 = llvm.call @malloc(%90) : (i64) -> !llvm.ptr<i8>
    %92 = llvm.bitcast %91 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %93 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %94 = llvm.insertvalue %92, %93[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.mlir.constant(0 : index) : i64
    %97 = llvm.insertvalue %96, %95[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.insertvalue %40, %97[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.insertvalue %87, %98[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%12 : i64)
  ^bb4(%100: i64):  // 2 preds: ^bb3, ^bb5
    %101 = llvm.icmp "slt" %100, %40 : i64
    llvm.cond_br %101, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %102 = llvm.getelementptr %92[%100] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %102 : !llvm.ptr<i64>
    %103 = llvm.add %100, %14  : i64
    llvm.br ^bb4(%103 : i64)
  ^bb6:  // pred: ^bb4
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
    %113 = llvm.getelementptr %112[%42] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %114 = llvm.ptrtoint %113 : !llvm.ptr<i64> to i64
    %115 = llvm.call @malloc(%114) : (i64) -> !llvm.ptr<i8>
    %116 = llvm.bitcast %115 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %117 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %118 = llvm.insertvalue %116, %117[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %119 = llvm.insertvalue %116, %118[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %120 = llvm.mlir.constant(0 : index) : i64
    %121 = llvm.insertvalue %120, %119[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.insertvalue %42, %121[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.insertvalue %111, %122[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%12 : i64)
  ^bb7(%124: i64):  // 2 preds: ^bb6, ^bb8
    %125 = llvm.icmp "slt" %124, %42 : i64
    llvm.cond_br %125, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %126 = llvm.getelementptr %116[%124] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %126 : !llvm.ptr<i64>
    %127 = llvm.add %124, %14  : i64
    llvm.br ^bb7(%127 : i64)
  ^bb9:  // pred: ^bb7
    %128 = llvm.mlir.constant(1 : index) : i64
    %129 = llvm.alloca %128 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %123, %129 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %130 = llvm.bitcast %129 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %131 = llvm.mlir.constant(1 : index) : i64
    %132 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %133 = llvm.insertvalue %131, %132[0] : !llvm.struct<(i64, ptr<i8>)> 
    %134 = llvm.insertvalue %130, %133[1] : !llvm.struct<(i64, ptr<i8>)> 
    %135 = llvm.mlir.constant(1 : index) : i64
    %136 = llvm.mlir.null : !llvm.ptr<i64>
    %137 = llvm.getelementptr %136[%44] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %138 = llvm.ptrtoint %137 : !llvm.ptr<i64> to i64
    %139 = llvm.call @malloc(%138) : (i64) -> !llvm.ptr<i8>
    %140 = llvm.bitcast %139 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %141 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %142 = llvm.insertvalue %140, %141[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.insertvalue %140, %142[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.mlir.constant(0 : index) : i64
    %145 = llvm.insertvalue %144, %143[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.insertvalue %44, %145[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.insertvalue %135, %146[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%12 : i64)
  ^bb10(%148: i64):  // 2 preds: ^bb9, ^bb11
    %149 = llvm.icmp "slt" %148, %44 : i64
    llvm.cond_br %149, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %150 = llvm.getelementptr %140[%148] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %150 : !llvm.ptr<i64>
    %151 = llvm.add %148, %14  : i64
    llvm.br ^bb10(%151 : i64)
  ^bb12:  // pred: ^bb10
    %152 = llvm.mlir.constant(1 : index) : i64
    %153 = llvm.alloca %152 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %147, %153 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %154 = llvm.bitcast %153 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %155 = llvm.mlir.constant(1 : index) : i64
    %156 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %157 = llvm.insertvalue %155, %156[0] : !llvm.struct<(i64, ptr<i8>)> 
    %158 = llvm.insertvalue %154, %157[1] : !llvm.struct<(i64, ptr<i8>)> 
    %159 = llvm.mlir.constant(1 : index) : i64
    %160 = llvm.mlir.null : !llvm.ptr<i64>
    %161 = llvm.getelementptr %160[%46] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %162 = llvm.ptrtoint %161 : !llvm.ptr<i64> to i64
    %163 = llvm.call @malloc(%162) : (i64) -> !llvm.ptr<i8>
    %164 = llvm.bitcast %163 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %165 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %166 = llvm.insertvalue %164, %165[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.insertvalue %164, %166[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %168 = llvm.mlir.constant(0 : index) : i64
    %169 = llvm.insertvalue %168, %167[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %170 = llvm.insertvalue %46, %169[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %171 = llvm.insertvalue %159, %170[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%12 : i64)
  ^bb13(%172: i64):  // 2 preds: ^bb12, ^bb14
    %173 = llvm.icmp "slt" %172, %46 : i64
    llvm.cond_br %173, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %174 = llvm.getelementptr %164[%172] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %174 : !llvm.ptr<i64>
    %175 = llvm.add %172, %14  : i64
    llvm.br ^bb13(%175 : i64)
  ^bb15:  // pred: ^bb13
    %176 = llvm.mlir.constant(1 : index) : i64
    %177 = llvm.alloca %176 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %171, %177 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %178 = llvm.bitcast %177 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %179 = llvm.mlir.constant(1 : index) : i64
    %180 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %181 = llvm.insertvalue %179, %180[0] : !llvm.struct<(i64, ptr<i8>)> 
    %182 = llvm.insertvalue %178, %181[1] : !llvm.struct<(i64, ptr<i8>)> 
    %183 = llvm.mlir.constant(1 : index) : i64
    %184 = llvm.mlir.null : !llvm.ptr<i64>
    %185 = llvm.getelementptr %184[%48] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %186 = llvm.ptrtoint %185 : !llvm.ptr<i64> to i64
    %187 = llvm.call @malloc(%186) : (i64) -> !llvm.ptr<i8>
    %188 = llvm.bitcast %187 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %189 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %190 = llvm.insertvalue %188, %189[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %191 = llvm.insertvalue %188, %190[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %192 = llvm.mlir.constant(0 : index) : i64
    %193 = llvm.insertvalue %192, %191[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %194 = llvm.insertvalue %48, %193[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %195 = llvm.insertvalue %183, %194[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%12 : i64)
  ^bb16(%196: i64):  // 2 preds: ^bb15, ^bb17
    %197 = llvm.icmp "slt" %196, %48 : i64
    llvm.cond_br %197, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %198 = llvm.getelementptr %188[%196] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %198 : !llvm.ptr<i64>
    %199 = llvm.add %196, %14  : i64
    llvm.br ^bb16(%199 : i64)
  ^bb18:  // pred: ^bb16
    %200 = llvm.mlir.constant(1 : index) : i64
    %201 = llvm.alloca %200 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %195, %201 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %202 = llvm.bitcast %201 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %203 = llvm.mlir.constant(1 : index) : i64
    %204 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %205 = llvm.insertvalue %203, %204[0] : !llvm.struct<(i64, ptr<i8>)> 
    %206 = llvm.insertvalue %202, %205[1] : !llvm.struct<(i64, ptr<i8>)> 
    %207 = llvm.mlir.constant(1 : index) : i64
    %208 = llvm.mlir.null : !llvm.ptr<i64>
    %209 = llvm.getelementptr %208[%50] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %210 = llvm.ptrtoint %209 : !llvm.ptr<i64> to i64
    %211 = llvm.call @malloc(%210) : (i64) -> !llvm.ptr<i8>
    %212 = llvm.bitcast %211 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %213 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %214 = llvm.insertvalue %212, %213[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %215 = llvm.insertvalue %212, %214[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %216 = llvm.mlir.constant(0 : index) : i64
    %217 = llvm.insertvalue %216, %215[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %218 = llvm.insertvalue %50, %217[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %219 = llvm.insertvalue %207, %218[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%12 : i64)
  ^bb19(%220: i64):  // 2 preds: ^bb18, ^bb20
    %221 = llvm.icmp "slt" %220, %50 : i64
    llvm.cond_br %221, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %222 = llvm.getelementptr %212[%220] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %222 : !llvm.ptr<i64>
    %223 = llvm.add %220, %14  : i64
    llvm.br ^bb19(%223 : i64)
  ^bb21:  // pred: ^bb19
    %224 = llvm.mlir.constant(1 : index) : i64
    %225 = llvm.alloca %224 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %219, %225 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %226 = llvm.bitcast %225 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %227 = llvm.mlir.constant(1 : index) : i64
    %228 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %229 = llvm.insertvalue %227, %228[0] : !llvm.struct<(i64, ptr<i8>)> 
    %230 = llvm.insertvalue %226, %229[1] : !llvm.struct<(i64, ptr<i8>)> 
    %231 = llvm.mlir.constant(1 : index) : i64
    %232 = llvm.mlir.null : !llvm.ptr<i64>
    %233 = llvm.getelementptr %232[%52] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %234 = llvm.ptrtoint %233 : !llvm.ptr<i64> to i64
    %235 = llvm.call @malloc(%234) : (i64) -> !llvm.ptr<i8>
    %236 = llvm.bitcast %235 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %237 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %238 = llvm.insertvalue %236, %237[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %239 = llvm.insertvalue %236, %238[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %240 = llvm.mlir.constant(0 : index) : i64
    %241 = llvm.insertvalue %240, %239[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %242 = llvm.insertvalue %52, %241[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %243 = llvm.insertvalue %231, %242[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%12 : i64)
  ^bb22(%244: i64):  // 2 preds: ^bb21, ^bb23
    %245 = llvm.icmp "slt" %244, %52 : i64
    llvm.cond_br %245, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %246 = llvm.getelementptr %236[%244] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %246 : !llvm.ptr<i64>
    %247 = llvm.add %244, %14  : i64
    llvm.br ^bb22(%247 : i64)
  ^bb24:  // pred: ^bb22
    %248 = llvm.mlir.constant(1 : index) : i64
    %249 = llvm.alloca %248 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %243, %249 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %250 = llvm.bitcast %249 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %251 = llvm.mlir.constant(1 : index) : i64
    %252 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %253 = llvm.insertvalue %251, %252[0] : !llvm.struct<(i64, ptr<i8>)> 
    %254 = llvm.insertvalue %250, %253[1] : !llvm.struct<(i64, ptr<i8>)> 
    %255 = llvm.mlir.constant(1 : index) : i64
    %256 = llvm.mlir.null : !llvm.ptr<i64>
    %257 = llvm.getelementptr %256[%54] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %258 = llvm.ptrtoint %257 : !llvm.ptr<i64> to i64
    %259 = llvm.call @malloc(%258) : (i64) -> !llvm.ptr<i8>
    %260 = llvm.bitcast %259 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %261 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %262 = llvm.insertvalue %260, %261[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %263 = llvm.insertvalue %260, %262[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %264 = llvm.mlir.constant(0 : index) : i64
    %265 = llvm.insertvalue %264, %263[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %266 = llvm.insertvalue %54, %265[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %267 = llvm.insertvalue %255, %266[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%12 : i64)
  ^bb25(%268: i64):  // 2 preds: ^bb24, ^bb26
    %269 = llvm.icmp "slt" %268, %54 : i64
    llvm.cond_br %269, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %270 = llvm.getelementptr %260[%268] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %270 : !llvm.ptr<i64>
    %271 = llvm.add %268, %14  : i64
    llvm.br ^bb25(%271 : i64)
  ^bb27:  // pred: ^bb25
    %272 = llvm.mlir.constant(1 : index) : i64
    %273 = llvm.alloca %272 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %267, %273 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %274 = llvm.bitcast %273 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %275 = llvm.mlir.constant(1 : index) : i64
    %276 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %277 = llvm.insertvalue %275, %276[0] : !llvm.struct<(i64, ptr<i8>)> 
    %278 = llvm.insertvalue %274, %277[1] : !llvm.struct<(i64, ptr<i8>)> 
    %279 = llvm.mlir.constant(1 : index) : i64
    %280 = llvm.mlir.null : !llvm.ptr<i64>
    %281 = llvm.getelementptr %280[%56] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %282 = llvm.ptrtoint %281 : !llvm.ptr<i64> to i64
    %283 = llvm.call @malloc(%282) : (i64) -> !llvm.ptr<i8>
    %284 = llvm.bitcast %283 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %285 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %286 = llvm.insertvalue %284, %285[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %287 = llvm.insertvalue %284, %286[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %288 = llvm.mlir.constant(0 : index) : i64
    %289 = llvm.insertvalue %288, %287[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %290 = llvm.insertvalue %56, %289[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %291 = llvm.insertvalue %279, %290[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb28(%12 : i64)
  ^bb28(%292: i64):  // 2 preds: ^bb27, ^bb29
    %293 = llvm.icmp "slt" %292, %56 : i64
    llvm.cond_br %293, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    %294 = llvm.getelementptr %284[%292] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %294 : !llvm.ptr<i64>
    %295 = llvm.add %292, %14  : i64
    llvm.br ^bb28(%295 : i64)
  ^bb30:  // pred: ^bb28
    %296 = llvm.mlir.constant(1 : index) : i64
    %297 = llvm.alloca %296 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %291, %297 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %298 = llvm.bitcast %297 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %299 = llvm.mlir.constant(1 : index) : i64
    %300 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %301 = llvm.insertvalue %299, %300[0] : !llvm.struct<(i64, ptr<i8>)> 
    %302 = llvm.insertvalue %298, %301[1] : !llvm.struct<(i64, ptr<i8>)> 
    %303 = llvm.mlir.constant(1 : index) : i64
    %304 = llvm.mlir.null : !llvm.ptr<i64>
    %305 = llvm.getelementptr %304[%58] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %306 = llvm.ptrtoint %305 : !llvm.ptr<i64> to i64
    %307 = llvm.call @malloc(%306) : (i64) -> !llvm.ptr<i8>
    %308 = llvm.bitcast %307 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %309 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %310 = llvm.insertvalue %308, %309[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %311 = llvm.insertvalue %308, %310[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %312 = llvm.mlir.constant(0 : index) : i64
    %313 = llvm.insertvalue %312, %311[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %314 = llvm.insertvalue %58, %313[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %315 = llvm.insertvalue %303, %314[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb31(%12 : i64)
  ^bb31(%316: i64):  // 2 preds: ^bb30, ^bb32
    %317 = llvm.icmp "slt" %316, %58 : i64
    llvm.cond_br %317, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %318 = llvm.getelementptr %308[%316] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %318 : !llvm.ptr<i64>
    %319 = llvm.add %316, %14  : i64
    llvm.br ^bb31(%319 : i64)
  ^bb33:  // pred: ^bb31
    %320 = llvm.mlir.constant(1 : index) : i64
    %321 = llvm.alloca %320 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %315, %321 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %322 = llvm.bitcast %321 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %323 = llvm.mlir.constant(1 : index) : i64
    %324 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %325 = llvm.insertvalue %323, %324[0] : !llvm.struct<(i64, ptr<i8>)> 
    %326 = llvm.insertvalue %322, %325[1] : !llvm.struct<(i64, ptr<i8>)> 
    %327 = llvm.mlir.constant(1 : index) : i64
    %328 = llvm.mlir.null : !llvm.ptr<i64>
    %329 = llvm.getelementptr %328[%60] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %330 = llvm.ptrtoint %329 : !llvm.ptr<i64> to i64
    %331 = llvm.call @malloc(%330) : (i64) -> !llvm.ptr<i8>
    %332 = llvm.bitcast %331 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %333 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %334 = llvm.insertvalue %332, %333[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %335 = llvm.insertvalue %332, %334[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %336 = llvm.mlir.constant(0 : index) : i64
    %337 = llvm.insertvalue %336, %335[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %338 = llvm.insertvalue %60, %337[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %339 = llvm.insertvalue %327, %338[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb34(%12 : i64)
  ^bb34(%340: i64):  // 2 preds: ^bb33, ^bb35
    %341 = llvm.icmp "slt" %340, %60 : i64
    llvm.cond_br %341, ^bb35, ^bb36
  ^bb35:  // pred: ^bb34
    %342 = llvm.getelementptr %332[%340] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %12, %342 : !llvm.ptr<i64>
    %343 = llvm.add %340, %14  : i64
    llvm.br ^bb34(%343 : i64)
  ^bb36:  // pred: ^bb34
    %344 = llvm.mlir.constant(1 : index) : i64
    %345 = llvm.alloca %344 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %339, %345 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %346 = llvm.bitcast %345 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %347 = llvm.mlir.constant(1 : index) : i64
    %348 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %349 = llvm.insertvalue %347, %348[0] : !llvm.struct<(i64, ptr<i8>)> 
    %350 = llvm.insertvalue %346, %349[1] : !llvm.struct<(i64, ptr<i8>)> 
    %351 = llvm.mlir.constant(1 : index) : i64
    %352 = llvm.mlir.null : !llvm.ptr<f64>
    %353 = llvm.getelementptr %352[%62] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %354 = llvm.ptrtoint %353 : !llvm.ptr<f64> to i64
    %355 = llvm.call @malloc(%354) : (i64) -> !llvm.ptr<i8>
    %356 = llvm.bitcast %355 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %357 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %358 = llvm.insertvalue %356, %357[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %359 = llvm.insertvalue %356, %358[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %360 = llvm.mlir.constant(0 : index) : i64
    %361 = llvm.insertvalue %360, %359[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %362 = llvm.insertvalue %62, %361[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %363 = llvm.insertvalue %351, %362[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb37(%12 : i64)
  ^bb37(%364: i64):  // 2 preds: ^bb36, ^bb38
    %365 = llvm.icmp "slt" %364, %62 : i64
    llvm.cond_br %365, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %366 = llvm.getelementptr %356[%364] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %366 : !llvm.ptr<f64>
    %367 = llvm.add %364, %14  : i64
    llvm.br ^bb37(%367 : i64)
  ^bb39:  // pred: ^bb37
    %368 = llvm.mlir.constant(1 : index) : i64
    %369 = llvm.alloca %368 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %363, %369 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %370 = llvm.bitcast %369 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %371 = llvm.mlir.constant(1 : index) : i64
    %372 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %373 = llvm.insertvalue %371, %372[0] : !llvm.struct<(i64, ptr<i8>)> 
    %374 = llvm.insertvalue %370, %373[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_3D_f64(%13, %14, %15, %14, %15, %14, %15, %83, %82, %107, %106, %131, %130, %155, %154, %179, %178, %203, %202, %227, %226, %251, %250, %275, %274, %299, %298, %323, %322, %347, %346, %371, %370, %13) {filename = "SPARSE_FILE_NAME1"} : (i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    llvm.call @comet_print_memref_i64(%83, %82) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%107, %106) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%131, %130) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%155, %154) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%179, %178) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%203, %202) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%227, %226) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%251, %250) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%275, %274) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%299, %298) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%323, %322) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_i64(%347, %346) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @comet_print_memref_f64(%371, %370) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_3D_f64(i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_3D_f64(i32, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @quick_sort(i64, !llvm.ptr<i8>, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_i64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
