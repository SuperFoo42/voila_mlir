#pragma once
#include "Logical.hpp"         // for Logical
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Or : public Logical
    {
        ASTNodeVariant mLhs, mRhs;

      public:
        Or(Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs) : Logical(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_or() const final;

        Or *as_or() final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &lhs() const { return mLhs; }

        [[nodiscard]] const ASTNodeVariant &rhs() const { return mRhs; }
    };
} // namespace voila::ast