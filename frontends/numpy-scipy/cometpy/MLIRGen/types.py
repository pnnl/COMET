DENSE = 0 
CSR = 1
COO = 2
CSC = 3
UNSUPPORTED_FORMAT = -1


class ShapedType:
    def __init__(self, shape, element_type, format):
        self.shape = shape
        self.element_type = element_type
        self.format = format

    
class TensorIndexType:
    def __init__(self):
        pass

    def __str__(self):
        return '!ta.index'

class MemrefType(ShapedType):

    def __init__(self, shape, element_type):
        super().__init__(shape, element_type, DENSE)

    def __str__(self):
        return "memref<{}x{}>".format("x".join([str(d) for d in self.shape]), self.element_type)
    
class TensorType(ShapedType):

    def __init__(self, shape, element_type):
        super().__init__(shape, element_type, DENSE)

    def __str__(self):
        return "tensor<{}x{}>".format("x".join([str(d) for d in self.shape]), self.element_type)


class TASparseTensorType(ShapedType):
    def __init__(self, shape, element_type, indices_type, format):
        super().__init__(shape, element_type, format)
        self.indices_type = indices_type
        if format == CSR: 
            self.sp_format = 'd, unk, cu, unk'
        elif format == COO:
            self.sp_format = 'cn, unk, s, unk'

    def __str__(self):
        return "!ta.sparse_tensor<{}, {}, {}, {}>".format(self.element_type, self.indices_type, "x".join(['?' for s in self.shape]), self.sp_format) 
