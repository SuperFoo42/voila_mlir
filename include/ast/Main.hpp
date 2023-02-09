#pragma once
#include "ASTNode.hpp" // for Location
#include "ast/Fun.hpp" // for Fun
#include <string>      // for string
#include <vector>      // for vector

namespace voila::ast
{
    class ASTVisitor;
    class Expression;
    class Statement;

    class Main : public Fun
    {
      public:
        Main(Location loc, std::vector<Expression> args, std::vector<Statement> exprs);

        Main() = default;
        Main(Main &) = default;
        Main(const Main &) = default;
        Main(Main &&) = default;

        Main &operator=(const Main &) = default;

        [[nodiscard]] bool is_main() const override;

        Main *as_main() override;

        [[nodiscard]] std::string type2string() const override;

        void visit(ASTVisitor &visitor) const override;
        void visit(ASTVisitor &visitor) override;

        using Fun::clone;
    };
} // namespace voila::ast
