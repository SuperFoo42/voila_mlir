#pragma once

#include "Const.hpp"

#include <utility>
#include "ASTVisitor.hpp"

namespace voila::ast {
    class StrConst : public Const {
    public:
        explicit StrConst(const Location loc, std::string val) : Const(loc), val{std::move(val)} {}

        [[nodiscard]] bool is_string() const final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &) override;

        const std::string val;
    };
} // namespace voila::ast