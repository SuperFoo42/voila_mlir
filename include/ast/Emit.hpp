#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class Emit : public IStatement
    {
      public:
        explicit Emit(Expression expr) : IStatement(), expr{std::move(expr)} {}

        [[nodiscard]] bool is_emit() const final;

        Emit *as_emit() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        Expression expr;
    };

} // namespace voila::ast