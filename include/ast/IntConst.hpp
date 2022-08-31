#pragma once

#include "Const.hpp"
#include "ast/ASTVisitor.hpp"

namespace voila::ast {
    class IntConst : public Const {

    public:
        explicit IntConst(Location loc, const std::intmax_t val) : Const(loc), val{val} {}

        [[nodiscard]] bool is_integer() const final;

        IntConst *as_integer() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &) override;

        const std::intmax_t val;
    };
} // namespace voila::ast