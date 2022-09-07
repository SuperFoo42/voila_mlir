#pragma once
#include "Const.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class FltConst : public Const
    {
      public:
        explicit FltConst(const Location loc, const double val) : Const(loc), val{val} {}

        [[nodiscard]] bool is_float() const final;

        FltConst *as_float() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &) override;

        const double val;
    };
} // namespace voila::ast