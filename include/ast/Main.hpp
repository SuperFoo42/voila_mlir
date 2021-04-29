#pragma once
#include "ASTNode.hpp"
#include "ASTVisitor.hpp"
#include "Statement.hpp"
#include "ast/Fun.hpp"
#include "ast/IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
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
    };
} // namespace voila::ast
