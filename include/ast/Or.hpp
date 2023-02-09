#pragma once
#include "Expression.hpp"      // for Expression
#include "Logical.hpp"         // for Logical
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class ASTVisitor;

    class Or : public Logical
    {
        Expression mLhs, mRhs;

      public:
        Or(Location loc, Expression lhs, Expression rhs) : Logical(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_or() const final;

        Or *as_or() final;

        void visit(ASTVisitor &visitor) final;
        void visit(ASTVisitor &visitor) const final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &lhs() const { return mLhs; }

        [[nodiscard]] const Expression &rhs() const { return mRhs; }
    };
} // namespace voila::ast