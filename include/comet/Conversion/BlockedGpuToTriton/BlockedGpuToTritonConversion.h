
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
namespace mlir {
namespace comet{

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter(MLIRContext *context);
private:
  MLIRContext *context;
};

class TritonConversionTarget : public ConversionTarget {

public:
  explicit TritonConversionTarget(MLIRContext &ctx,
                                     TritonTypeConverter &typeConverter);
};

}
}