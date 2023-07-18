//===- MarkerUtils.h - Marker Utils --*- C++ -*-===//
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTICE: The source code has been modified for integration with COMET.


#include "comet/Conversion/Utils/MarkerUtils.h"

#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace comet {

struct VectorTransforms {
  static const StringLiteral kVectorTransformMarker;
};
const StringLiteral VectorTransforms::kVectorTransformMarker =
    "__internal_vector_transform__";

StringRef getFusedMarker() { return "fused_numprocs_ge_numiters"; }

StringRef getWorkgroupKTiledMarker() { return "workgroup_k_tiled"; }

StringRef getWorkgroupL1TileMarker() { return "workgroup_l1_tile"; }

StringRef getWorkgroupMemoryMarker() { return "workgroup_memory"; }

StringRef getWorkgroupNumItemsGENumItersMarker() {
  return "workgroup_numprocs_ge_numiters";
}

StringRef getWorkgroupMemoryNumItemsGENumItersMarker() {
  return "workgroup_memory_numprocs_ge_numiters";
}

StringRef getCopyToWorkgroupMemoryMarker() {
  return "copy_to_workgroup_memory";
}

StringRef getTileReductionMarker() { return "tile_reduction"; }

StringRef getVectorizeMarker() { return "vectorize"; }

StringRef getDeleteMarker() { return "delete"; }

StringRef getMarkerOrNull(Operation *op) {
  StringAttr attr = op->getAttrOfType<StringAttr>(
     "__internal_linalg_transform__");
  if (!attr) return "";
  return attr.getValue();
}

bool hasMarker(Operation *op, ArrayRef<StringRef> marker) {
  StringAttr attr = op->getAttrOfType<StringAttr>(
      "__internal_linalg_transform__");
  return attr && (marker.empty() ||
                  llvm::any_of(marker, [&attr](StringRef markerValue) {
                    return attr.getValue() == markerValue;
                  }));
}

void setMarker(Operation *op, StringRef marker) {
  op->setAttr(
                "__internal_linalg_transform__",
              StringAttr::get(op->getContext(), marker));
}

}  // namespace comet
}  // namespace mlir
