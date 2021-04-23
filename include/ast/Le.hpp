#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Le : public BinaryOP<Expression>, public Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_le() const final;

        Le *as_le() final;
        void print(std::ostream &ostream) const final;

      protected:
        void checkArgs(Expression &lhs, Expression &rhs) final;
    };
} // namespace voila::ast