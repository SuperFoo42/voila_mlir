#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"
namespace voila::ast
{
    class Ref : public IExpression
    {
      public:
        explicit Ref(Location loc, Expression ref);

        [[nodiscard]] bool is_reference() const final;

        [[nodiscard]] std::string type2string() const override;

        const Ref *as_reference() const final;

        void print(std::ostream &o) const final;

        Expression ref;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast