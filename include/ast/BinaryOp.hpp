#pragma once
#include "ASTNodeVariant.hpp"
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for make_shared, shared_ptr
#include <string>              // for string

namespace voila::ast
{
    template <class BinOp> class BinaryOp : public AbstractASTNode<BinOp>
    {
        ASTNodeVariant mLhs, mRhs;

      public:
        BinaryOp(Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs)
            : AbstractASTNode<BinOp>(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)}
        {
        }

        void print_impl(std::ostream &) const {};

        [[nodiscard]] const ASTNodeVariant &lhs() const { return mLhs; }

        [[nodiscard]] const ASTNodeVariant &rhs() const { return mRhs; }

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
        {
            auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                           [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
            return std::make_shared<BinOp>(this->loc, std::visit(cloneVisitor, mLhs), std::visit(cloneVisitor, mRhs));
        }
    };
} // namespace voila::ast