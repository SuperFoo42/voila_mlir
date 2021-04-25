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
        [[nodiscard]] virtual std::string type2string() const = 0;
        virtual void visit(ASTVisitor &visitor) const;
        virtual void visit(ASTVisitor &visitor);
        virtual ~ASTNode() = default;

        [[nodiscard]] virtual bool is_expr() const;

        [[nodiscard]] virtual bool is_stmt() const;

        [[nodiscard]] virtual bool is_function_definition() const;

        [[nodiscard]] virtual bool is_main() const;

        virtual Fun *as_function_definition();

        virtual Main *as_main();
    };
} // namespace voila::ast
