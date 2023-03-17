#pragma once
#include "ASTNode.hpp" // for Location
#include "ast/Fun.hpp" // for Fun
#include <string>      // for string
#include <vector>      // for vector

namespace voila::ast
{
    class Expression;
    class Statement;

    class Main : public Fun
    {
      public:

        Main(Location loc, std::vector<ASTNodeVariant> args, std::vector<ASTNodeVariant> exprs);

        Main() = default;
        Main(Main &) = default;
        Main(const Main &) = default;
        Main(Main &&) = default;

        Main &operator=(const Main &) = default;

        [[nodiscard]] bool is_main() const override;

        Main *as_main() override;

        [[nodiscard]] std::string type2string() const override;

        using Fun::clone;
    };
} // namespace voila::ast
