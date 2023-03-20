#pragma once
#include "ASTNode.hpp" // for Location
#include "ast/Fun.hpp" // for Fun
#include <string>      // for string
#include <vector>      // for vector
#include "range/v3/range/concepts.hpp"

namespace voila::ast
{
    class Main : public Fun
    {
      public:
        Main(Location loc, ranges::input_range auto && args, ranges::input_range auto && exprs)
            : Fun(loc, "main", args, exprs) {}

        [[nodiscard]] std::string type2string_impl() const final { return "main function"; };
    };
} // namespace voila::ast
