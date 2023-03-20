#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class ASTVisitor;

    class BooleanConst : public Const
    {
      public:
        explicit BooleanConst(const Location loc, const bool val) : AbstractASTNode<BooleanConst>(loc), val{val} {}

        [[nodiscard]] std::string type2string_impl() const { return "bool"; }
        void print_impl(std::ostream &ostream) const { ostream << std::boolalpha << val << std::noboolalpha; };

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const bool val;
    };
} // namespace voila::ast