#include "LayoutConversionCost.h"

#include <cstdint>

#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project

#define DEBUG_TYPE "layout-conversion-cost"

namespace mlir {
namespace heir {

using Cost = int64_t;
using tensor_ext::LayoutAttr;

// An unset value of a permutation as it's being built up.
static constexpr int kUnset = -1;

// Return the first output index not mapped to by the partial permutation.
int64_t getMinUnusedTarget(llvm::ArrayRef<int64_t> perm) {
  std::vector<int64_t> unmappedOutputsVector(perm.size());
  std::iota(unmappedOutputsVector.begin(), unmappedOutputsVector.end(), 0);
  std::set<int64_t> unmappedOutputs(unmappedOutputsVector.begin(),
                                    unmappedOutputsVector.end());
  for (int64_t target : perm) {
    if (target != kUnset) {
      unmappedOutputs.erase(target);
    }
  }

  if (unmappedOutputs.empty()) {
    return -1;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Unmapped outputs: ";
    for (int64_t i : unmappedOutputs) {
      llvm::dbgs() << i << " ";
    }
    llvm::dbgs() << "\n";
  });

  return *unmappedOutputs.begin();
}

// Return the first unused input index not mapped from by the partial
// permutation.
int64_t getMinUnusedInput(llvm::ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < perm.size(); ++i) {
    if (perm[i] == kUnset) return i;
  }
  return -1;
}

Cost createPermuation(Value value, int64_t slots, LayoutAttr fromLayout,
                      LayoutAttr toLayout) {
  Type dataSemanticType = value.getType();
  SmallVector<int64_t> permutation(slots, kUnset);

  // see
  // lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.cpp::ConvertConvertLayout
  // for a full explanation of the algorithm
  int64_t minUnusedTarget = 0;
  int64_t minUnusedInput = 0;
  while (minUnusedInput != -1) {
    IndexTupleConsumer evaluateNextIndex =
        [&](const std::vector<int64_t> &indices) {
          SmallVector<int64_t> fromResults;
          SmallVector<int64_t> toResults;
          evaluateStatic(fromLayout.getMap(), indices, fromResults);
          evaluateStatic(toLayout.getMap(), indices, toResults);
          int64_t input =
              (minUnusedInput + fromResults[fromResults.size() - 1]) % slots;
          int64_t output =
              (minUnusedTarget + toResults[toResults.size() - 1]) % slots;
          permutation[input] = output;
        };

    SmallVector<int64_t> dataSemanticShape;
    if (auto tensorTy = dyn_cast<RankedTensorType>(dataSemanticType)) {
      dataSemanticShape = SmallVector<int64_t>(tensorTy.getShape());
    } else {
      // assumed to be a scalar
      dataSemanticShape = {1};
    }

    LLVM_DEBUG(llvm::dbgs() << "dataSemanticShape: \n"; {
      for (int64_t val : dataSemanticShape) {
        llvm::dbgs() << val << ' ';
      };
    } llvm::dbgs() << "\n";);

    iterateIndices(dataSemanticShape, evaluateNextIndex);
    minUnusedTarget = getMinUnusedTarget(permutation);
    minUnusedInput = getMinUnusedInput(permutation);
  }

  LLVM_DEBUG(llvm::dbgs() << "my permutation: \n"; {
    for (int64_t val : permutation) {
      llvm::dbgs() << val << ' ';
    };
  } llvm::dbgs() << "\n";);
  return 5;
}

Cost layoutAnalysis(Value value, int64_t slots, LayoutAttr fromLayout,
                    LayoutAttr toLayout) {
  return createPermuation(value, slots, fromLayout, toLayout);
}

}  // namespace heir
}  // namespace mlir
