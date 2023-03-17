#pragma once
#include "Aggregation.hpp"     // for Aggregation
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/ASTNodeVariant.hpp"
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class AggrAvg : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) final;

        [[nodiscard]] bool is_aggr_avg() const final;

        AggrAvg *as_aggr_avg() final;

        [[nodiscard]] std::string type2string() const final;
    };
} // namespace voila::ast