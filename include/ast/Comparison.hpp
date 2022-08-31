#pragma once
#include <utility>

#include "IExpression.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Comparison : public IExpression
    {
      public:
        Comparison(const Location loc, Expression lhs, Expression rhs) : IExpression(loc), lhs{std::move(lhs)}, rhs{std::move(rhs)} {}
        [[nodiscard]] bool is_comparison() const final;

        Comparison *as_comparison() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &ostream) const final;

        Expression lhs;
        Expression rhs;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final;
    };
} // namespace voila::ast