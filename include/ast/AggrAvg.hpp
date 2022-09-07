#pragma once
#include "ASTVisitor.hpp"
#include "Aggregation.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class AggrAvg : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Aggregation::clone<AggrAvg>(vmap);
        }

        [[nodiscard]] bool is_aggr_avg() const final;

        AggrAvg *as_aggr_avg() final;

        [[nodiscard]] std::string type2string() const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast