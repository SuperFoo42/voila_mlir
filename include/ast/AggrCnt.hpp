#pragma once
#include "Aggregation.hpp"     // for Aggregation
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class AggrCnt : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) final;

        [[nodiscard]] bool is_aggr_cnt() const final;

        AggrCnt *as_aggr_cnt() final;

        [[nodiscard]] std::string type2string() const final;
    };
} // namespace voila::ast