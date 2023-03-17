#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <utility>              // for move
#include "Logical.hpp"          // for Logical
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class And : public Logical
    {
        ASTNodeVariant mLhs, mRhs;

      public:
        And(const Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs) :
            Logical(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_and() const final;

        And *as_and() final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] [[nodiscard]] const ASTNodeVariant &lhs() const {
            return mLhs;
        }

        [[nodiscard]] [[nodiscard]] const ASTNodeVariant &rhs() const {
            return mRhs;
        }
    };
} // namespace voila::ast