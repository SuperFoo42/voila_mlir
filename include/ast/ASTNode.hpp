#pragma once
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

namespace voila::ast
{
    class ASTVisitor;
    class Fun;
    class Main;

    class ASTNode
    {
      public:
        virtual void print(std::ostream &) const = 0;
        virtual std::string type2string() const = 0;
        virtual void visit(ASTVisitor &visitor);
        virtual ~ASTNode() = default;

        virtual bool is_expr() const
        {
            return false;
        }

        virtual bool is_stmt() const
        {
            return false;
        }

        virtual bool is_function_definition() const
        {
            return false;
        }

        virtual bool is_main() const
        {
            return false;
        }

        virtual Fun *as_function_definition()
        {
            return nullptr;
        }

        virtual Main *as_main()
        {
            return nullptr;
        }
    };
} // namespace voila::ast