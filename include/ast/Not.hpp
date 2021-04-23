#pragma once
#include "Expression.hpp"
#include "Logical.hpp"
#include "UnaryOP.hpp"
namespace voila::ast
{
    class Not : public UnaryOP<Expression>, public Logical
    {
      public:
        using UnaryOP::param;
        explicit Not(Expression expr) : UnaryOP<Expression>(expr)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_not() const final;

        Not *as_not() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast