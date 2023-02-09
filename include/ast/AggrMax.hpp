#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include "Aggregation.hpp"      // for Aggregation
#include "Expression.hpp"       // for Expression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class ASTVisitor;

    class AggrMax : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Aggregation::clone<AggrMax>(vmap);
        }

        [[nodiscard]] bool is_aggr_max() const final;

        AggrMax *as_aggr_max() final;

        [[nodiscard]] std::string type2string() const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast