#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"
#include "Logical.hpp"
namespace voila::ast
{
    class And : public BinaryOP<Expression>, public Logical
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_and() const final;

        And *as_and() final;
        void print(std::ostream &ostream) const final;

      protected:
        void checkArgs(Expression &lhs, Expression &rhs) final;
    };
} // namespace voila::ast