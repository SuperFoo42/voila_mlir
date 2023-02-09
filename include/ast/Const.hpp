#pragma once
#include <string>           // for string
#include "IExpression.hpp"  // for IExpression
#include "ast/ASTNode.hpp"  // for Location

namespace voila::ast
{
    class Const : public IExpression
    {
      public:
        explicit Const(Location loc) : IExpression(loc) {}
        ~Const() override = default;

        [[nodiscard]] bool is_const() const final;

        Const *as_const() final;

        [[nodiscard]] std::string type2string() const override;
    };
} // namespace voila::ast