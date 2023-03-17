#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include "Aggregation.hpp"      // for Aggregation
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class AggrSum : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) final;

        [[nodiscard]] bool is_aggr_sum() const final;

        AggrSum *as_aggr_sum() final;

        [[nodiscard]] std::string type2string() const final;
    };
} // namespace voila::ast