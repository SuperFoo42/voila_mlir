#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include "Const.hpp"            // for Const
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class ASTVisitor;

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

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const bool val;
    };
} // namespace voila::ast