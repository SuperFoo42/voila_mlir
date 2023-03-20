#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class ASTVisitor;

    class StrConst : public Const
    {
      public:
        explicit StrConst(const Location loc, std::string val) : AbstractASTNode<StrConst>(loc), val{std::move(val)} {}

        [[nodiscard]] std::string type2string_impl() const { return "string"; }

        void print_impl(std::ostream &ostream) const { ostream << "\"" << val << "\""; }

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &) override;

        const std::string val;
    };
} // namespace voila::ast