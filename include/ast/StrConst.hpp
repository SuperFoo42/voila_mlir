#pragma once
#include "Const.hpp"

#include <utility>
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class StrConst : public Const
    {
      public:
        explicit StrConst(std::string val) : Const(), val{std::move(val)} {}

        [[nodiscard]] bool is_string() const final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        const std::string val;
    };
} // namespace voila::ast