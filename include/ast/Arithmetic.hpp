#pragma once
#include "IExpression.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Arithmetic : public IExpression
    {
      public:
        Arithmetic(Location loc, Expression lhs, Expression rhs);
        ~Arithmetic() override = default;

        [[nodiscard]] bool is_arithmetic() const final;

        Arithmetic *as_arithmetic() final;

        [[nodiscard]] std::string type2string() const override;
        void print(std::ostream &ostream) const final;

        Expression lhs, rhs;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final;
    };
} // namespace voila::ast