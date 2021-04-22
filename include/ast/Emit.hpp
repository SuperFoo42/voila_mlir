#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <vector>

namespace voila::ast
{
    class Emit : IStatement
    {
      public:
        Emit(const Expression &expr, const Expression &pred) : IStatement(), expr{expr}, pred{pred} {}

        bool is_emit() const final
        {
            return true;
        }

        Emit *as_emit() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "emit";
        }

        Expression expr;
        Expression pred;
    };

} // namespace voila::ast