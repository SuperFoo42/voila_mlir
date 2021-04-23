#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class FunctionCall : public IStatement
    {
      public:
        FunctionCall(std::string fun, std::vector<std::string> args);

        [[nodiscard]] bool is_function_call() const final;

        FunctionCall *as_function_call() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        std::string fun;
        std::vector<std::string> args;
    };
} // namespace voila::ast