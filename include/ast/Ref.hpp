#pragma once
#include "Expression.hpp"
#include "IExpression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Ref : public IExpression
    {
      public:
        explicit Ref(const std::string &var);

        [[nodiscard]] bool is_reference() const final;

        [[nodiscard]] std::string type2string() const override;

        Ref *as_reference() final;

        void print(std::ostream &o) const final;

        Expression ref;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast