#pragma once
#include "Const.hpp"
#include "ast/ASTVisitor.hpp"

namespace voila::ast
{
    class IntConst : public Const
    {
      public:
        explicit IntConst(const std::intmax_t val) : Const(), val{val} {}

        [[nodiscard]] bool is_integer() const final;

        IntConst *as_integer() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        const std::intmax_t val;
    };
} // namespace voila::ast