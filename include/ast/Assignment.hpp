#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <vector>

namespace voila::ast
{
    class Assign : IStatement
    {
      public:
        Assign(const std::string &dest, const Expression &expr) : IStatement(), dest{dest}, expr{expr}, pred{pred}
        {
            // TODO: find dest variable and look for conflicts
        }

        bool is_assignment() const final
        {
            return true;
        }

        Assign *as_assignment() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "assignment";
        }

        std::string dest;
        Expression expr;
        Expression pred;
    };
} // namespace voila::ast