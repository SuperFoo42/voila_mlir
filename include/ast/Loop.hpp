#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"
#include "Statement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class Loop : public IStatement
    {
      public:
        Loop(Expression pred, std::vector<Statement> stms) : IStatement(), pred{std::move(pred)}, stms{std::move(stms)}
        {
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_loop() const final;

        Loop *as_loop() final;
        void print(std::ostream &ostream) const final;

        Expression pred;
        std::vector<Statement> stms;
        // TODO
        // CrossingVariables crossing_variables;
    };

} // namespace voila::ast