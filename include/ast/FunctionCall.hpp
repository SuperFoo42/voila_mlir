#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <vector>

namespace voila::ast
{
    class FunctionCall : IStatement
    {
      public:
        FunctionCall(const std::string fun, std::vector<Expression> args) : IStatement(), fun{fun}, args{args}
        {
            // TODO: lookup function definition and check if all arguments match
        }

        bool is_function_call() const final
        {
            return true;
        }

        FunctionCall *as_function_call() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "function call";
        }

        std::string fun;
        std::vector<Expression> args;
    };
} // namespace voila::ast