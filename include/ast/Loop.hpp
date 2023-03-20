#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector
#include "ASTNodeVariant.hpp"
#include "range/v3/all.hpp"
namespace voila::ast
{

    // TODO: fix this
    class Loop : public AbstractASTNode<Loop>
    {
        ASTNodeVariant mPred;
        std::vector<ASTNodeVariant> mStms;

      public:
        Loop(const Location loc, ASTNodeVariant pred, ranges::input_range auto && stms)
            : AbstractASTNode<Loop>(loc), mPred{std::move(pred)}, mStms{ranges::to<std::vector>(stms)}
        {
        }

        [[nodiscard]] std::string type2string_impl() const { return "loop"; };

        void print_impl(std::ostream &) const {};

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &pred() const { return mPred; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &stmts() const { return mStms; }
    };

} // namespace voila::ast