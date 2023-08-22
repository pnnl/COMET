module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(1.700000e+00 : f64) : f64
    %3 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %4 = llvm.mlir.constant(10 : index) : i64
    %5 = llvm.mlir.constant(9 : index) : i64
    %6 = llvm.mlir.constant(8 : index) : i64
    %7 = llvm.mlir.constant(7 : index) : i64
    %8 = llvm.mlir.constant(6 : index) : i64
    %9 = llvm.mlir.constant(5 : index) : i64
    %10 = llvm.mlir.constant(4 : index) : i64
    %11 = llvm.mlir.constant(3 : index) : i64
    %12 = llvm.mlir.constant(2 : index) : i64
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
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
    llvm.call @read_input_sizes_2D_f64(%14, %0, %15, %1, %15, %33, %32, %13) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) -> ()
    %37 = llvm.getelementptr %22[%0] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %38 = llvm.load %37 : !llvm.ptr<i64>
    %39 = llvm.getelementptr %22[%1] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %40 = llvm.load %39 : !llvm.ptr<i64>
    %41 = llvm.getelementptr %22[%12] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %42 = llvm.load %41 : !llvm.ptr<i64>
    %43 = llvm.getelementptr %22[%11] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %44 = llvm.load %43 : !llvm.ptr<i64>
    %45 = llvm.getelementptr %22[%10] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %46 = llvm.load %45 : !llvm.ptr<i64>
    %47 = llvm.getelementptr %22[%9] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %48 = llvm.load %47 : !llvm.ptr<i64>
    %49 = llvm.getelementptr %22[%8] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %50 = llvm.load %49 : !llvm.ptr<i64>
    %51 = llvm.getelementptr %22[%7] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %52 = llvm.load %51 : !llvm.ptr<i64>
    %53 = llvm.getelementptr %22[%6] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %54 = llvm.load %53 : !llvm.ptr<i64>
    %55 = llvm.getelementptr %22[%5] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %56 = llvm.load %55 : !llvm.ptr<i64>
    %57 = llvm.getelementptr %22[%4] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
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
    %75 = llvm.add %72, %1  : i64
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
    %99 = llvm.add %96, %1  : i64
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
    %123 = llvm.add %120, %1  : i64
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
    %147 = llvm.add %144, %1  : i64
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
    %171 = llvm.add %168, %1  : i64
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
    %195 = llvm.add %192, %1  : i64
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
    %219 = llvm.add %216, %1  : i64
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
    %243 = llvm.add %240, %1  : i64
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
    llvm.store %3, %266 : !llvm.ptr<f64>
    %267 = llvm.add %264, %1  : i64
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
    llvm.call @read_input_2D_f64(%14, %0, %15, %1, %15, %79, %78, %103, %102, %127, %126, %151, %150, %175, %174, %199, %198, %223, %222, %247, %246, %271, %270, %13) {filename = "SPARSE_FILE_NAME0"} : (i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) -> ()
    %275 = llvm.mlir.constant(1 : index) : i64
    %276 = llvm.mlir.null : !llvm.ptr<f64>
    %277 = llvm.getelementptr %276[%58] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %278 = llvm.ptrtoint %277 : !llvm.ptr<f64> to i64
    %279 = llvm.mlir.constant(32 : index) : i64
    %280 = llvm.add %278, %279  : i64
    %281 = llvm.call @malloc(%280) : (i64) -> !llvm.ptr<i8>
    %282 = llvm.bitcast %281 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %283 = llvm.ptrtoint %282 : !llvm.ptr<f64> to i64
    %284 = llvm.mlir.constant(1 : index) : i64
    %285 = llvm.sub %279, %284  : i64
    %286 = llvm.add %283, %285  : i64
    %287 = llvm.urem %286, %279  : i64
    %288 = llvm.sub %286, %287  : i64
    %289 = llvm.inttoptr %288 : i64 to !llvm.ptr<f64>
    %290 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %291 = llvm.insertvalue %282, %290[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %292 = llvm.insertvalue %289, %291[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %293 = llvm.mlir.constant(0 : index) : i64
    %294 = llvm.insertvalue %293, %292[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %295 = llvm.insertvalue %58, %294[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %296 = llvm.insertvalue %275, %295[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %297 = llvm.mlir.constant(1 : index) : i64
    %298 = llvm.mlir.null : !llvm.ptr<f64>
    %299 = llvm.getelementptr %298[%56] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %300 = llvm.ptrtoint %299 : !llvm.ptr<f64> to i64
    %301 = llvm.mlir.constant(32 : index) : i64
    %302 = llvm.add %300, %301  : i64
    %303 = llvm.call @malloc(%302) : (i64) -> !llvm.ptr<i8>
    %304 = llvm.bitcast %303 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %305 = llvm.ptrtoint %304 : !llvm.ptr<f64> to i64
    %306 = llvm.mlir.constant(1 : index) : i64
    %307 = llvm.sub %301, %306  : i64
    %308 = llvm.add %305, %307  : i64
    %309 = llvm.urem %308, %301  : i64
    %310 = llvm.sub %308, %309  : i64
    %311 = llvm.inttoptr %310 : i64 to !llvm.ptr<f64>
    %312 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %313 = llvm.insertvalue %304, %312[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %314 = llvm.insertvalue %311, %313[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %315 = llvm.mlir.constant(0 : index) : i64
    %316 = llvm.insertvalue %315, %314[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %317 = llvm.insertvalue %56, %316[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    %318 = llvm.insertvalue %297, %317[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb28(%0 : i64)
  ^bb28(%319: i64):  // 2 preds: ^bb27, ^bb29
    %320 = llvm.icmp "slt" %319, %58 : i64
    llvm.cond_br %320, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    %321 = llvm.getelementptr %289[%319] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %2, %321 : !llvm.ptr<f64>
    %322 = llvm.add %319, %1  : i64
    llvm.br ^bb28(%322 : i64)
  ^bb30:  // pred: ^bb28
    llvm.br ^bb31(%0 : i64)
  ^bb31(%323: i64):  // 2 preds: ^bb30, ^bb32
    %324 = llvm.icmp "slt" %323, %56 : i64
    llvm.cond_br %324, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %325 = llvm.getelementptr %311[%323] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %3, %325 : !llvm.ptr<f64>
    %326 = llvm.add %323, %1  : i64
    llvm.br ^bb31(%326 : i64)
  ^bb33:  // pred: ^bb31
    %327 = llvm.getelementptr %64[%0] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %328 = llvm.load %327 : !llvm.ptr<i64>
    llvm.br ^bb34(%0 : i64)
  ^bb34(%329: i64):  // 2 preds: ^bb33, ^bb38
    %330 = llvm.icmp "slt" %329, %328 : i64
    llvm.cond_br %330, ^bb35, ^bb39
  ^bb35:  // pred: ^bb34
    %331 = llvm.add %329, %1  : i64
    %332 = llvm.getelementptr %160[%329] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %333 = llvm.load %332 : !llvm.ptr<i64>
    %334 = llvm.getelementptr %160[%331] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %335 = llvm.load %334 : !llvm.ptr<i64>
    llvm.br ^bb36(%333 : i64)
  ^bb36(%336: i64):  // 2 preds: ^bb35, ^bb37
    %337 = llvm.icmp "slt" %336, %335 : i64
    llvm.cond_br %337, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    %338 = llvm.getelementptr %184[%336] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %339 = llvm.load %338 : !llvm.ptr<i64>
    %340 = llvm.getelementptr %256[%336] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %341 = llvm.load %340 : !llvm.ptr<f64>
    %342 = llvm.getelementptr %289[%339] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %343 = llvm.load %342 : !llvm.ptr<f64>
    %344 = llvm.getelementptr %311[%329] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %345 = llvm.load %344 : !llvm.ptr<f64>
    %346 = llvm.fmul %341, %343  : f64
    %347 = llvm.fadd %345, %346  : f64
    %348 = llvm.getelementptr %311[%329] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %347, %348 : !llvm.ptr<f64>
    %349 = llvm.add %336, %1  : i64
    llvm.br ^bb36(%349 : i64)
  ^bb38:  // pred: ^bb36
    %350 = llvm.add %329, %1  : i64
    llvm.br ^bb34(%350 : i64)
  ^bb39:  // pred: ^bb34
    %351 = llvm.mlir.constant(1 : index) : i64
    %352 = llvm.alloca %351 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %318, %352 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    %353 = llvm.bitcast %352 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %354 = llvm.mlir.constant(1 : index) : i64
    %355 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %356 = llvm.insertvalue %354, %355[0] : !llvm.struct<(i64, ptr<i8>)> 
    %357 = llvm.insertvalue %353, %356[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @comet_print_memref_f64(%354, %353) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @read_input_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @read_input_sizes_2D_f64(i32, i64, i64, i64, i64, i64, !llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @comet_sort_index(i64, !llvm.ptr<i8>, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @comet_print_memref_f64(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}
