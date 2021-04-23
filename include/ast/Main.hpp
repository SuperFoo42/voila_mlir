#pragma once
#include "ASTNode.hpp"
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
        Main(std::vector<std::string> args, std::vector<Statement> exprs);

        Main() = default;
        Main(Main &) = default;
        Main(const Main &) = default;
        Main(Main &&) = default;

        Main &operator=(const Main &) = default;

        [[nodiscard]] bool is_main() const final;

        Main *as_main() final;

        [[nodiscard]] std::string type2string() const final;
    };
} // namespace voila::ast
