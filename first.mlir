module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1.700000e+00 : f64) : f64
    %4 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(10 : index) : i64
    %6 = llvm.mlir.constant(9 : index) : i64
    %7 = llvm.mlir.constant(8 : index) : i64
    %8 = llvm.mlir.constant(7 : index) : i64
    %9 = llvm.mlir.constant(6 : index) : i64
    %10 = llvm.mlir.constant(5 : index) : i64
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.constant(3 : index) : i64
    %14 = llvm.mlir.constant(2 : index) : i64
    %15 = llvm.mlir.constant(-1 : index) : i64
    %16 = llvm.mlir.constant(13 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.null : !llvm.ptr<i64>
    %19 = llvm.getelementptr %18[13] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
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
    llvm.call @read_input_sizes_2D_f64(%12, %14, %15, %13, %15, %33, %32, %11) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %37 = llvm.getelementptr %22[%0] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %38 = llvm.load %37 : !llvm.ptr<i64>
    %39 = llvm.getelementptr %22[%2] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %40 = llvm.load %39 : !llvm.ptr<i64>
    %41 = llvm.getelementptr %22[%14] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %42 = llvm.load %41 : !llvm.ptr<i64>
    %43 = llvm.getelementptr %22[%13] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %44 = llvm.load %43 : !llvm.ptr<i64>
    %45 = llvm.getelementptr %22[%1] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %46 = llvm.load %45 : !llvm.ptr<i64>
    %47 = llvm.getelementptr %22[%10] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %48 = llvm.load %47 : !llvm.ptr<i64>
    %49 = llvm.getelementptr %22[%9] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %50 = llvm.load %49 : !llvm.ptr<i64>
    %51 = llvm.getelementptr %22[%8] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %52 = llvm.load %51 : !llvm.ptr<i64>
    %53 = llvm.getelementptr %22[%7] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %54 = llvm.load %53 : !llvm.ptr<i64>
    %55 = llvm.getelementptr %22[%6] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %56 = llvm.load %55 : !llvm.ptr<i64>
    %57 = llvm.getelementptr %22[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %58 = llvm.load %57 : !llvm.ptr<i64>
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.mlir.null : !llvm.ptr<i64>
    %61 = llvm.getelementptr %60[%38] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %62 = llvm.ptrtoint %61 : !llvm.ptr<i64> to i64
    %63 = llvm.call @malloc(%62) : (i64) -> !llvm.ptr<i8>
    %64 = llvm.bitcast %63 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %65 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %66 = llvm.insertvalue %64, %65[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.insertvalue %68, %67[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.insertvalue %38, %69[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %71 = llvm.insertvalue %59, %70[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%0 : i64)
  ^bb1(%72: i64):  // 2 preds: ^bb0, ^bb2
    %73 = llvm.icmp "slt" %72, %38 : i64
    llvm.cond_br %73, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %74 = llvm.getelementptr %64[%72] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %74 : !llvm.ptr<i64>
    %75 = llvm.add %72, %2  : i64
    llvm.br ^bb1(%75 : i64)
  ^bb3:  // pred: ^bb1
    %76 = llvm.mlir.constant(1 : index) : i64
    %77 = llvm.alloca %76 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %71, %77 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %78 = llvm.bitcast %77 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %79 = llvm.mlir.constant(1 : index) : i64
    %80 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %81 = llvm.insertvalue %79, %80[0] : !llvm.struct<(i64, ptr<i8>)> 
    %82 = llvm.insertvalue %78, %81[1] : !llvm.struct<(i64, ptr<i8>)> 
    %83 = llvm.mlir.constant(1 : index) : i64
    %84 = llvm.mlir.null : !llvm.ptr<i64>
    %85 = llvm.getelementptr %84[%40] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %86 = llvm.ptrtoint %85 : !llvm.ptr<i64> to i64
    %87 = llvm.call @malloc(%86) : (i64) -> !llvm.ptr<i8>
    %88 = llvm.bitcast %87 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %89 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %90 = llvm.insertvalue %88, %89[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %91 = llvm.insertvalue %88, %90[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %92 = llvm.mlir.constant(0 : index) : i64
    %93 = llvm.insertvalue %92, %91[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %94 = llvm.insertvalue %40, %93[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.insertvalue %83, %94[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%0 : i64)
  ^bb4(%96: i64):  // 2 preds: ^bb3, ^bb5
    %97 = llvm.icmp "slt" %96, %40 : i64
    llvm.cond_br %97, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %98 = llvm.getelementptr %88[%96] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %98 : !llvm.ptr<i64>
    %99 = llvm.add %96, %2  : i64
    llvm.br ^bb4(%99 : i64)
  ^bb6:  // pred: ^bb4
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.alloca %100 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %95, %101 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %102 = llvm.bitcast %101 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %103 = llvm.mlir.constant(1 : index) : i64
    %104 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %105 = llvm.insertvalue %103, %104[0] : !llvm.struct<(i64, ptr<i8>)> 
    %106 = llvm.insertvalue %102, %105[1] : !llvm.struct<(i64, ptr<i8>)> 
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.mlir.null : !llvm.ptr<i64>
    %109 = llvm.getelementptr %108[%42] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %110 = llvm.ptrtoint %109 : !llvm.ptr<i64> to i64
    %111 = llvm.call @malloc(%110) : (i64) -> !llvm.ptr<i8>
    %112 = llvm.bitcast %111 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %113 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %114 = llvm.insertvalue %112, %113[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %115 = llvm.insertvalue %112, %114[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %116 = llvm.mlir.constant(0 : index) : i64
    %117 = llvm.insertvalue %116, %115[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %118 = llvm.insertvalue %42, %117[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %119 = llvm.insertvalue %107, %118[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb7(%0 : i64)
  ^bb7(%120: i64):  // 2 preds: ^bb6, ^bb8
    %121 = llvm.icmp "slt" %120, %42 : i64
    llvm.cond_br %121, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %122 = llvm.getelementptr %112[%120] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %122 : !llvm.ptr<i64>
    %123 = llvm.add %120, %2  : i64
    llvm.br ^bb7(%123 : i64)
  ^bb9:  // pred: ^bb7
    %124 = llvm.mlir.constant(1 : index) : i64
    %125 = llvm.alloca %124 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %119, %125 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %126 = llvm.bitcast %125 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %127 = llvm.mlir.constant(1 : index) : i64
    %128 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %129 = llvm.insertvalue %127, %128[0] : !llvm.struct<(i64, ptr<i8>)> 
    %130 = llvm.insertvalue %126, %129[1] : !llvm.struct<(i64, ptr<i8>)> 
    %131 = llvm.mlir.constant(1 : index) : i64
    %132 = llvm.mlir.null : !llvm.ptr<i64>
    %133 = llvm.getelementptr %132[%44] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %134 = llvm.ptrtoint %133 : !llvm.ptr<i64> to i64
    %135 = llvm.call @malloc(%134) : (i64) -> !llvm.ptr<i8>
    %136 = llvm.bitcast %135 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %137 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %138 = llvm.insertvalue %136, %137[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %139 = llvm.insertvalue %136, %138[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %140 = llvm.mlir.constant(0 : index) : i64
    %141 = llvm.insertvalue %140, %139[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.insertvalue %44, %141[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.insertvalue %131, %142[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb10(%0 : i64)
  ^bb10(%144: i64):  // 2 preds: ^bb9, ^bb11
    %145 = llvm.icmp "slt" %144, %44 : i64
    llvm.cond_br %145, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %146 = llvm.getelementptr %136[%144] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %146 : !llvm.ptr<i64>
    %147 = llvm.add %144, %2  : i64
    llvm.br ^bb10(%147 : i64)
  ^bb12:  // pred: ^bb10
    %148 = llvm.mlir.constant(1 : index) : i64
    %149 = llvm.alloca %148 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %143, %149 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %150 = llvm.bitcast %149 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %153 = llvm.insertvalue %151, %152[0] : !llvm.struct<(i64, ptr<i8>)> 
    %154 = llvm.insertvalue %150, %153[1] : !llvm.struct<(i64, ptr<i8>)> 
    %155 = llvm.mlir.constant(1 : index) : i64
    %156 = llvm.mlir.null : !llvm.ptr<i64>
    %157 = llvm.getelementptr %156[%46] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %158 = llvm.ptrtoint %157 : !llvm.ptr<i64> to i64
    %159 = llvm.call @malloc(%158) : (i64) -> !llvm.ptr<i8>
    %160 = llvm.bitcast %159 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %161 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %162 = llvm.insertvalue %160, %161[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %163 = llvm.insertvalue %160, %162[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %164 = llvm.mlir.constant(0 : index) : i64
    %165 = llvm.insertvalue %164, %163[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %166 = llvm.insertvalue %46, %165[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %167 = llvm.insertvalue %155, %166[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb13(%0 : i64)
  ^bb13(%168: i64):  // 2 preds: ^bb12, ^bb14
    %169 = llvm.icmp "slt" %168, %46 : i64
    llvm.cond_br %169, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %170 = llvm.getelementptr %160[%168] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %170 : !llvm.ptr<i64>
    %171 = llvm.add %168, %2  : i64
    llvm.br ^bb13(%171 : i64)
  ^bb15:  // pred: ^bb13
    %172 = llvm.mlir.constant(1 : index) : i64
    %173 = llvm.alloca %172 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %167, %173 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %174 = llvm.bitcast %173 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %175 = llvm.mlir.constant(1 : index) : i64
    %176 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %177 = llvm.insertvalue %175, %176[0] : !llvm.struct<(i64, ptr<i8>)> 
    %178 = llvm.insertvalue %174, %177[1] : !llvm.struct<(i64, ptr<i8>)> 
    %179 = llvm.mlir.constant(1 : index) : i64
    %180 = llvm.mlir.null : !llvm.ptr<i64>
    %181 = llvm.getelementptr %180[%48] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %182 = llvm.ptrtoint %181 : !llvm.ptr<i64> to i64
    %183 = llvm.call @malloc(%182) : (i64) -> !llvm.ptr<i8>
    %184 = llvm.bitcast %183 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %185 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %186 = llvm.insertvalue %184, %185[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %187 = llvm.insertvalue %184, %186[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %188 = llvm.mlir.constant(0 : index) : i64
    %189 = llvm.insertvalue %188, %187[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %190 = llvm.insertvalue %48, %189[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %191 = llvm.insertvalue %179, %190[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb16(%0 : i64)
  ^bb16(%192: i64):  // 2 preds: ^bb15, ^bb17
    %193 = llvm.icmp "slt" %192, %48 : i64
    llvm.cond_br %193, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %194 = llvm.getelementptr %184[%192] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %194 : !llvm.ptr<i64>
    %195 = llvm.add %192, %2  : i64
    llvm.br ^bb16(%195 : i64)
  ^bb18:  // pred: ^bb16
    %196 = llvm.mlir.constant(1 : index) : i64
    %197 = llvm.alloca %196 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %191, %197 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %198 = llvm.bitcast %197 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %199 = llvm.mlir.constant(1 : index) : i64
    %200 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %201 = llvm.insertvalue %199, %200[0] : !llvm.struct<(i64, ptr<i8>)> 
    %202 = llvm.insertvalue %198, %201[1] : !llvm.struct<(i64, ptr<i8>)> 
    %203 = llvm.mlir.constant(1 : index) : i64
    %204 = llvm.mlir.null : !llvm.ptr<i64>
    %205 = llvm.getelementptr %204[%50] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %206 = llvm.ptrtoint %205 : !llvm.ptr<i64> to i64
    %207 = llvm.call @malloc(%206) : (i64) -> !llvm.ptr<i8>
    %208 = llvm.bitcast %207 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %209 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %210 = llvm.insertvalue %208, %209[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %211 = llvm.insertvalue %208, %210[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %212 = llvm.mlir.constant(0 : index) : i64
    %213 = llvm.insertvalue %212, %211[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %214 = llvm.insertvalue %50, %213[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %215 = llvm.insertvalue %203, %214[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb19(%0 : i64)
  ^bb19(%216: i64):  // 2 preds: ^bb18, ^bb20
    %217 = llvm.icmp "slt" %216, %50 : i64
    llvm.cond_br %217, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %218 = llvm.getelementptr %208[%216] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %218 : !llvm.ptr<i64>
    %219 = llvm.add %216, %2  : i64
    llvm.br ^bb19(%219 : i64)
  ^bb21:  // pred: ^bb19
    %220 = llvm.mlir.constant(1 : index) : i64
    %221 = llvm.alloca %220 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %215, %221 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %222 = llvm.bitcast %221 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %223 = llvm.mlir.constant(1 : index) : i64
    %224 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %225 = llvm.insertvalue %223, %224[0] : !llvm.struct<(i64, ptr<i8>)> 
    %226 = llvm.insertvalue %222, %225[1] : !llvm.struct<(i64, ptr<i8>)> 
    %227 = llvm.mlir.constant(1 : index) : i64
    %228 = llvm.mlir.null : !llvm.ptr<i64>
    %229 = llvm.getelementptr %228[%52] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %230 = llvm.ptrtoint %229 : !llvm.ptr<i64> to i64
    %231 = llvm.call @malloc(%230) : (i64) -> !llvm.ptr<i8>
    %232 = llvm.bitcast %231 : !llvm.ptr<i8> to !llvm.ptr<i64>
    %233 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %234 = llvm.insertvalue %232, %233[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %235 = llvm.insertvalue %232, %234[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %236 = llvm.mlir.constant(0 : index) : i64
    %237 = llvm.insertvalue %236, %235[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %238 = llvm.insertvalue %52, %237[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %239 = llvm.insertvalue %227, %238[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb22(%0 : i64)
  ^bb22(%240: i64):  // 2 preds: ^bb21, ^bb23
    %241 = llvm.icmp "slt" %240, %52 : i64
    llvm.cond_br %241, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %242 = llvm.getelementptr %232[%240] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %0, %242 : !llvm.ptr<i64>
    %243 = llvm.add %240, %2  : i64
    llvm.br ^bb22(%243 : i64)
  ^bb24:  // pred: ^bb22
    %244 = llvm.mlir.constant(1 : index) : i64
    %245 = llvm.alloca %244 x !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %239, %245 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %246 = llvm.bitcast %245 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %247 = llvm.mlir.constant(1 : index) : i64
    %248 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %249 = llvm.insertvalue %247, %248[0] : !llvm.struct<(i64, ptr<i8>)> 
    %250 = llvm.insertvalue %246, %249[1] : !llvm.struct<(i64, ptr<i8>)> 
    %251 = llvm.mlir.constant(1 : index) : i64
    %252 = llvm.mlir.null : !llvm.ptr<f64>
    %253 = llvm.getelementptr %252[%54] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %254 = llvm.ptrtoint %253 : !llvm.ptr<f64> to i64
    %255 = llvm.call @malloc(%254) : (i64) -> !llvm.ptr<i8>
    %256 = llvm.bitcast %255 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %257 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %258 = llvm.insertvalue %256, %257[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %259 = llvm.insertvalue %256, %258[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %260 = llvm.mlir.constant(0 : index) : i64
    %261 = llvm.insertvalue %260, %259[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %262 = llvm.insertvalue %54, %261[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %263 = llvm.insertvalue %251, %262[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb25(%0 : i64)
  ^bb25(%264: i64):  // 2 preds: ^bb24, ^bb26
    %265 = llvm.icmp "slt" %264, %54 : i64
    llvm.cond_br %265, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %266 = llvm.getelementptr %256[%264] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %4, %266 : !llvm.ptr<f64>
    %267 = llvm.add %264, %2  : i64
    llvm.br ^bb25(%267 : i64)
  ^bb27:  // pred: ^bb25
    %268 = llvm.mlir.constant(1 : index) : i64
    %269 = llvm.alloca %268 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %263, %269 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %270 = llvm.bitcast %269 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %271 = llvm.mlir.constant(1 : index) : i64
    %272 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %273 = llvm.insertvalue %271, %272[0] : !llvm.struct<(i64, ptr<i8>)> 
    %274 = llvm.insertvalue %270, %273[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @read_input_2D_f64(%12, %14, %15, %13, %15, %79, %78, %103, %102, %127, %126, %151, %150, %175, %174, %199, %198, %223, %222, %247, %246, %271, %270, %11) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    %275 = llvm.mlir.constant(4 : index) : i64
    %276 = llvm.mlir.constant(1 : index) : i64
    %277 = llvm.mul %56, %275  : i64
    %278 = llvm.mlir.null : !llvm.ptr<f64>
    %279 = llvm.getelementptr %278[%277] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %280 = llvm.ptrtoint %279 : !llvm.ptr<f64> to i64
    %281 = llvm.mlir.constant(32 : index) : i64
    %282 = llvm.add %280, %281  : i64
    %283 = llvm.call @malloc(%282) : (i64) -> !llvm.ptr<i8>
    %284 = llvm.bitcast %283 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %285 = llvm.ptrtoint %284 : !llvm.ptr<f64> to i64
    %286 = llvm.mlir.constant(1 : index) : i64
    %287 = llvm.sub %281, %286  : i64
    %288 = llvm.add %285, %287  : i64
    %289 = llvm.urem %288, %281  : i64
    %290 = llvm.sub %288, %289  : i64
    %291 = llvm.inttoptr %290 : i64 to !llvm.ptr<f64>
    %292 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %293 = llvm.insertvalue %284, %292[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %294 = llvm.insertvalue %291, %293[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %295 = llvm.mlir.constant(0 : index) : i64
    %296 = llvm.insertvalue %295, %294[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %297 = llvm.insertvalue %275, %296[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %298 = llvm.insertvalue %56, %297[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %299 = llvm.insertvalue %56, %298[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %300 = llvm.insertvalue %276, %299[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %301 = llvm.mlir.constant(4 : index) : i64
    %302 = llvm.mlir.constant(1 : index) : i64
    %303 = llvm.mul %58, %301  : i64
    %304 = llvm.mlir.null : !llvm.ptr<f64>
    %305 = llvm.getelementptr %304[%303] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %306 = llvm.ptrtoint %305 : !llvm.ptr<f64> to i64
    %307 = llvm.mlir.constant(32 : index) : i64
    %308 = llvm.add %306, %307  : i64
    %309 = llvm.call @malloc(%308) : (i64) -> !llvm.ptr<i8>
    %310 = llvm.bitcast %309 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %311 = llvm.ptrtoint %310 : !llvm.ptr<f64> to i64
    %312 = llvm.mlir.constant(1 : index) : i64
    %313 = llvm.sub %307, %312  : i64
    %314 = llvm.add %311, %313  : i64
    %315 = llvm.urem %314, %307  : i64
    %316 = llvm.sub %314, %315  : i64
    %317 = llvm.inttoptr %316 : i64 to !llvm.ptr<f64>
    %318 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %319 = llvm.insertvalue %310, %318[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %320 = llvm.insertvalue %317, %319[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %321 = llvm.mlir.constant(0 : index) : i64
    %322 = llvm.insertvalue %321, %320[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %323 = llvm.insertvalue %301, %322[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %324 = llvm.insertvalue %58, %323[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %325 = llvm.insertvalue %58, %324[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %326 = llvm.insertvalue %302, %325[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb28(%0 : i64)
  ^bb28(%327: i64):  // 2 preds: ^bb27, ^bb32
    %328 = llvm.icmp "slt" %327, %1 : i64
    llvm.cond_br %328, ^bb29, ^bb33
  ^bb29:  // pred: ^bb28
    llvm.br ^bb30(%0 : i64)
  ^bb30(%329: i64):  // 2 preds: ^bb29, ^bb31
    %330 = llvm.icmp "slt" %329, %56 : i64
    llvm.cond_br %330, ^bb31, ^bb32
  ^bb31:  // pred: ^bb30
    %331 = llvm.mul %327, %56  : i64
    %332 = llvm.add %331, %329  : i64
    %333 = llvm.getelementptr %291[%332] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %3, %333 : !llvm.ptr<f64>
    %334 = llvm.add %329, %2  : i64
    llvm.br ^bb30(%334 : i64)
  ^bb32:  // pred: ^bb30
    %335 = llvm.add %327, %2  : i64
    llvm.br ^bb28(%335 : i64)
  ^bb33:  // pred: ^bb28
    llvm.br ^bb34(%0 : i64)
  ^bb34(%336: i64):  // 2 preds: ^bb33, ^bb38
    %337 = llvm.icmp "slt" %336, %1 : i64
    llvm.cond_br %337, ^bb35, ^bb39
  ^bb35:  // pred: ^bb34
    llvm.br ^bb36(%0 : i64)
  ^bb36(%338: i64):  // 2 preds: ^bb35, ^bb37
    %339 = llvm.icmp "slt" %338, %58 : i64
    llvm.cond_br %339, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    %340 = llvm.mul %336, %58  : i64
    %341 = llvm.add %340, %338  : i64
    %342 = llvm.getelementptr %317[%341] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %4, %342 : !llvm.ptr<f64>
    %343 = llvm.add %338, %2  : i64
    llvm.br ^bb36(%343 : i64)
  ^bb38:  // pred: ^bb36
    %344 = llvm.add %336, %2  : i64
    llvm.br ^bb34(%344 : i64)
  ^bb39:  // pred: ^bb34
    llvm.br ^bb40(%0 : i64)
  ^bb40(%345: i64):  // 2 preds: ^bb39, ^bb47
    %346 = llvm.icmp "slt" %345, %1 : i64
    llvm.cond_br %346, ^bb41, ^bb48
  ^bb41:  // pred: ^bb40
    %347 = llvm.getelementptr %64[%0] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %348 = llvm.load %347 : !llvm.ptr<i64>
    %349 = llvm.getelementptr %64[%2] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %350 = llvm.load %349 : !llvm.ptr<i64>
    llvm.br ^bb42(%348 : i64)
  ^bb42(%351: i64):  // 2 preds: ^bb41, ^bb46
    %352 = llvm.icmp "slt" %351, %350 : i64
    llvm.cond_br %352, ^bb43, ^bb47
  ^bb43:  // pred: ^bb42
    %353 = llvm.getelementptr %88[%351] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %354 = llvm.load %353 : !llvm.ptr<i64>
    %355 = llvm.getelementptr %136[%351] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %356 = llvm.load %355 : !llvm.ptr<i64>
    llvm.br ^bb44(%0 : i64)
  ^bb44(%357: i64):  // 2 preds: ^bb43, ^bb45
    %358 = llvm.icmp "slt" %357, %2 : i64
    llvm.cond_br %358, ^bb45, ^bb46
  ^bb45:  // pred: ^bb44
    %359 = llvm.mul %345, %56  : i64
    %360 = llvm.add %359, %354  : i64
    %361 = llvm.getelementptr %291[%360] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %362 = llvm.load %361 : !llvm.ptr<f64>
    %363 = llvm.getelementptr %256[%351] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %364 = llvm.load %363 : !llvm.ptr<f64>
    %365 = llvm.mul %345, %58  : i64
    %366 = llvm.add %365, %356  : i64
    %367 = llvm.getelementptr %317[%366] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %368 = llvm.load %367 : !llvm.ptr<f64>
    %369 = llvm.fmul %362, %364  : f64
    %370 = llvm.fadd %368, %369  : f64
    %371 = llvm.mul %345, %58  : i64
    %372 = llvm.add %371, %356  : i64
    %373 = llvm.getelementptr %317[%372] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %370, %373 : !llvm.ptr<f64>
    %374 = llvm.add %357, %2  : i64
    llvm.br ^bb44(%374 : i64)
  ^bb46:  // pred: ^bb44
    %375 = llvm.add %351, %2  : i64
    llvm.br ^bb42(%375 : i64)
  ^bb47:  // pred: ^bb42
    %376 = llvm.add %345, %2  : i64
    llvm.br ^bb40(%376 : i64)
  ^bb48:  // pred: ^bb40
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @quick_sort(i64, !llvm.ptr<i8>, i64) attributes {sym_visibility = "private"}
}
