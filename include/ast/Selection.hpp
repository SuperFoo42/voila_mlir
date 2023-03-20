#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Selection : public AbstractASTNode<Selection>
    {
        ASTNodeVariant mParam;
        ASTNodeVariant mPred;

      public:
        explicit Selection(const Location loc, ASTNodeVariant expr, ASTNodeVariant pred)
            : AbstractASTNode<Selection>(loc), mParam(std::move(expr)), mPred(std::move(pred))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string_impl() const { return "selection"; }
        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &param() const { return mParam; }

        [[nodiscard]] const ASTNodeVariant &pred() const { return mPred; }
    };
} // namespace voila::ast