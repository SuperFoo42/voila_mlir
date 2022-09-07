#pragma once
#include "ASTVisitor.hpp"
#include "Aggregation.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class AggrSum : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Aggregation::clone<AggrSum>(vmap);
        }

        [[nodiscard]] bool is_aggr_sum() const final;

        AggrSum *as_aggr_sum() final;

        [[nodiscard]] std::string type2string() const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast