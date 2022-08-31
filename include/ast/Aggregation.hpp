#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>

namespace voila::ast
{
    class Aggregation : public IExpression
    {
      public:
        Aggregation(const Location loc, Expression col) : IExpression(loc), src{std::move(col)}, groups(std::nullopt) {}
        Aggregation(const Location loc, Expression col, Expression groups) :
            IExpression(loc), src{std::move(col)}, groups(std::move(groups))
        {
        }
        ~Aggregation() override = default;

        [[nodiscard]] bool is_aggr() const final;

        Aggregation *as_aggr() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &) const final {}

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        Expression src;
        std::optional<Expression> groups;
    };
} // namespace voila::ast