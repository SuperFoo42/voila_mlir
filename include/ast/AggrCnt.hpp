#pragma once
#include "Aggregation.hpp"     // for Aggregation
#include "Expression.hpp"      // for Expression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class ASTVisitor;

    class AggrCnt : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Aggregation::clone<AggrCnt>(vmap);
        }

        [[nodiscard]] bool is_aggr_cnt() const final;

        AggrCnt *as_aggr_cnt() final;

        [[nodiscard]] std::string type2string() const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast