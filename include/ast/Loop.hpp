#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"
#include "Statement.hpp"

#include <vector>

namespace voila::ast
{
    class Loop : IStatement
    {
      public:
        Loop(const Expression &pred, const std::vector<Statement> &stms) : IStatement(), pred{pred}, stms{stms} {}

        std::string type2string() const final
        {
            return "loop";
        }

        bool is_loop() const final
        {
            return true;
        }

        Loop *as_loop() final
        {
            return this;
        }

        Expression pred;
        std::vector<Statement> stms;
        // TODO
        // CrossingVariables crossing_variables;
    };

} // namespace voila::ast