#
# Copyright 2022 Battelle Memorial Institute
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions 
# and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import re
from collections import OrderedDict
from typing import Sequence


class AliasMap(OrderedDict):
    def __setitem__(self, name, type):
        type = Type.find(type, aliases=self)
        super().__setitem__(name, type)


class Type:
    _subtypes = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self})"

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self.__class__ is other.__class__ and str(self) == str(other)

    def __init_subclass__(cls):
        Type._subtypes.append(cls)

    def to_pretty_string(self):
        return str(self)

    @staticmethod
    def find(text: str, aliases: AliasMap = None):
        if isinstance(text, Type):
            return text
        if aliases is not None and text[:1] == "#":
            if alias := aliases.get(text[1:]):
                return alias

        for klass in Type._subtypes:
            result = klass.parse(text, aliases=aliases)
            if result is not None:
                return result

        raise TypeError(f"Unknown type: {text}")


class IndexType(Type):
    def __init__(self):
        pass

    def __str__(self):
        return "index"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if text == "index":
            return IndexType()


class FloatType(Type):
    _patt = re.compile(r"^f(\d+)$")

    def __init__(self, num: int):
        self.num = num

    def __str__(self):
        return f"f{self.num}"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            return FloatType(int(m.group(1)))


class IntType(Type):
    _patt = re.compile(r"^i(\d+)$")

    def __init__(self, num: int):
        self.num = num

    def __str__(self):
        return f"i{self.num}"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            return IntType(int(m.group(1)))


class MemrefType(Type):
    _patt = re.compile(r"^memref<\s*((?:(?:[^,])*x)*)([^, ]+)\s*>$")

    def __init__(self, shape: Sequence[int], value_type: Type):
        shape = tuple(shape)
        if not isinstance(value_type, Type):
            raise TypeError(f"value_type must be a Type, not {type(value_type)}")
        elif not all(isinstance(dim, int) for dim in shape):
            raise TypeError(f"shape must be a sequence of ints, not {type(shape)}")
        self.shape = shape
        self.value_type = value_type

    def __str__(self):
        shape_string = "x".join("?" if dim == -1 else str(dim) for dim in self.shape)
        return f"memref<{shape_string}x{self.value_type}>"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            dim_strings = m.group(1).split("x")[:-1]
            if dim_strings == ["*"]:
                raise NotImplementedError(f"Unranked memrefs not currently supported.")
            elif (
                any(not s.isdigit() for s in dim_strings if s != "?")
                or len(dim_strings) == 0
            ):
                raise ValueError(f"{repr(text)} does not have a valid shape.")
            shape = [int(dim) if dim.isdigit() else -1 for dim in dim_strings]
            value_type = Type.find(m.group(2), aliases=aliases)
            return MemrefType(shape, value_type)


class TensorType(Type):
    _patt = re.compile(r"^tensor<\s*((?:(?:[^,])*x)*)([^, ]+)\s*>$")

    def __init__(self, shape: Sequence[int], value_type: Type):
        shape = tuple(shape)
        if not isinstance(value_type, Type):
            raise TypeError(f"value_type must be a Type, not {type(value_type)}")
        elif not all(isinstance(dim, int) for dim in shape):
            raise TypeError(f"shape must be a sequence of ints, not {type(shape)}")
        self.shape = shape
        self.value_type = value_type

    def __str__(self):
        shape_string = "x".join("?" if dim == -1 else str(dim) for dim in self.shape)
        return f"tensor<{shape_string}x{self.value_type}>"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            dim_strings = m.group(1).split("x")[:-1]
            if dim_strings == ["*"]:
                raise NotImplementedError(f"Unranked tensors not currently supported.")
            elif (
                any(not s.isdigit() for s in dim_strings if s != "?")
                or len(dim_strings) == 0
            ):
                raise ValueError(f"{repr(text)} does not have a valid shape.")
            shape = [int(dim) if dim.isdigit() else -1 for dim in dim_strings]
            value_type = Type.find(m.group(2), aliases=aliases)
            return TensorType(shape, value_type)


class SparseTensorType(Type):
    _patt = re.compile(r"^tensor<\s*((?:(?:.)*x)*)(.+),\s*(#.+)\s*>$")

    def __init__(
        self, shape: Sequence[int], value_type: Type, encoding: "SparseEncodingType"
    ):
        shape = tuple(shape)
        if not isinstance(value_type, Type):
            raise TypeError(f"value_type must be a Type, not {type(value_type)}")
        elif not isinstance(encoding, SparseEncodingType):
            raise TypeError(
                f"encoding must be a SparseEncodingType, not {type(encoding)}"
            )
        elif not all(isinstance(dim, int) for dim in shape):
            raise TypeError(f"shape must be a sequence of ints, not {type(shape)}")
        self.shape = shape
        self.value_type = value_type
        self.encoding = encoding

    def __str__(self):
        shape_string = "x".join("?" if dim == -1 else str(dim) for dim in self.shape)
        return f"tensor<{shape_string}x{self.value_type}, {self.encoding}>"

    def to_short_string(self):
        ret = []
        if self.encoding.rank == 2:
            ret.append("matrix")
            ret.append("csc" if self.encoding.ordering == [1, 0] else "csr")
        elif self.encoding.rank == 1:
            ret.append("vector")
        else:
            raise ValueError(f"Invalid rank: {self.encoding.rank}")
        ret.append(str(self.value_type))
        ret.append(
            f"p{self.encoding.pointer_bit_width}i{self.encoding.index_bit_width}"
        )
        return "_".join(ret)

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            dim_strings = m.group(1).split("x")[:-1]
            if dim_strings == ["*"]:
                raise NotImplementedError(f"Unranked tensors not currently supported.")
            elif (
                any(not s.isdigit() for s in dim_strings if s != "?")
                or len(dim_strings) == 0
            ):
                raise ValueError(f"{repr(text)} does not have a valid shape.")
            shape = [int(dim) if dim.isdigit() else -1 for dim in dim_strings]

            value_type = Type.find(m.group(2), aliases=aliases)
            encoding = Type.find(m.group(3), aliases=aliases)

            return SparseTensorType(shape, value_type, encoding)


class SparseEncodingType(Type):
    _patt = re.compile(
        r"^#sparse_tensor.encoding<\{"
        r"\s*,?\s*(?:dimLevelType\s*=\s*\[(?P<levels>.+)\])?"
        r"\s*,?\s*(?:dimOrdering\s*=\s*affine_map<(?P<ordering>.+)>)?"
        r"\s*,?\s*(?:pointerBitWidth\s*=\s*(?P<pointer>\d+))?"
        r"\s*,?\s*(?:indexBitWidth\s*=\s*(?P<index>\d+))?"
        r"\s*,?\s*\}>$"
    )

    def __init__(self, levels, ordering=None, pointer_bit_width=64, index_bit_width=64):
        if levels is None:
            if ordering is not None:
                raise TypeError("Cannot provide ordering without levels")
        self.levels = levels
        self.ordering = ordering
        self.pointer_bit_width = pointer_bit_width
        self.index_bit_width = index_bit_width

    @property
    def rank(self):
        if self.levels is None and self.ordering is None:
            return -1
        return len(self.levels)

    def __str__(self):
        return self.to_pretty_string(multiline=False)

    def to_pretty_string(self, multiline=True):
        internals = []
        if self.levels is not None:
            lvl_str = ", ".join(f'"{lvl}"' for lvl in self.levels)
            internals.append(f"dimLevelType = [ {lvl_str} ]")
            if self.ordering is not None:
                lhs = [f"d{i}" for i in range(len(self.levels))]
                rhs = [lhs[idx] for idx in self.ordering]
                internals.append(
                    f"dimOrdering = affine_map<({', '.join(lhs)}) -> ({', '.join(rhs)})>"
                )
        internals.append(f"pointerBitWidth = {self.pointer_bit_width}")
        internals.append(f"indexBitWidth = {self.index_bit_width}")
        if multiline:
            internals = ",\n    ".join(internals)
            return f"#sparse_tensor.encoding<{{\n    {internals}\n}}>"
        else:
            return f"#sparse_tensor.encoding<{{ {', '.join(internals)} }}>"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            if levels := m["levels"]:
                levels = [lvl.strip().strip("\"'") for lvl in levels.split(",")]
            if ordering := m["ordering"]:
                lhs, rhs = ordering.split("->")
                lhs = [x.strip() for x in lhs.strip().strip("()").split(",")]
                rhs = [x.strip() for x in rhs.strip().strip("()").split(",")]
                ordering = [lhs.index(x) for x in rhs]
            pointer_bit_width = int(m["pointer"])
            index_bit_width = int(m["index"])
            return SparseEncodingType(
                levels, ordering, pointer_bit_width, index_bit_width
            )


class LlvmPtrType(Type):
    _patt = re.compile(r"^!llvm\.ptr<\s*(.+)\s*>$")

    def __init__(self, internal_type: Type):
        if not isinstance(internal_type, Type):
            raise TypeError(f"internal_type must be a Type, not {type(internal_type)}")
        self.internal_type = internal_type

    def __str__(self):
        return f"!llvm.ptr<{self.internal_type}>"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            internal_type = Type.find(m.group(1), aliases=aliases)
            return LlvmPtrType(internal_type)


class AffineMap(Type):
    _patt = re.compile(r"^affine_map<([^>]+)>$")

    def __init__(self, text_within_angle_brackets):
        self.text_within_angle_brackets = text_within_angle_brackets

    def __str__(self):
        return f"affine_map<{self.text_within_angle_brackets}>"

    @classmethod
    def parse(cls, text: str, aliases: AliasMap = None):
        if m := cls._patt.match(text):
            return AffineMap(m.group(1))
