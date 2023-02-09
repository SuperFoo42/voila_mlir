#pragma once

#include "Const.hpp"           // for Const
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