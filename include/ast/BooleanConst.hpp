#pragma once
#include "Const.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class BooleanConst : public Const
    {
      public:
        explicit BooleanConst(const Location loc, const bool val) : Const(loc), val{val} {}

        [[nodiscard]] bool is_bool() const final;

        BooleanConst *as_bool() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        const bool val;
    };
} // namespace voila::ast