#pragma once
#include "ASTNode.hpp"
#include "Statement.hpp"

#include <vector>

namespace voila::ast
{
    class Fun : ASTNode
    {
      public:
        Fun(const std::string &fun, const std::vector<std::string> &args, const std::vector<Statement> &exprs) :
            ASTNode(), name{fun}, args{args}, body{exprs}
        {
            // TODO: check function, deduce argument types and register function
        }

        bool is_function_definition() const final
        {
            return true;
        }

        Fun *as_function_definition() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "function definition";
        }

        std::string name;
        std::vector<std::string> args;
        std::vector<Statement> body;
    };
} // namespace voila::ast