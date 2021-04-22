#pragma once
#include "ASTNode.hpp"
#include "Statement.hpp"
#include "ast/IStatement.hpp"

#include <vector>

namespace voila::ast
{
    class Main : ASTNode
    {
      public:
        Main(const std::vector<std::string> &args, const std::vector<Statement> &exprs) :
            ASTNode(), args{args}, exprs{exprs}
        {
            // TODO register as entry point and check args + exprs
        }

        bool is_main() const final
        {
            return true;
        }

        Main *as_main() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "main function";
        }

        std::vector<std::string> args;
        std::vector<Statement> exprs;
    };
} // namespace voila::ast
