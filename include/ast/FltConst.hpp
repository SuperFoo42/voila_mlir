#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include "Const.hpp"            // for Const
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{class ASTVisitor;

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