#include "ASTNode.hpp"
#pragma once

namespace voila::ast
{
    class Load : public AbstractASTNode<Load>
    {
        ASTNodeVariant mSrc, mDest, mMask;

      public:
        Load(const Location loc, ASTNodeVariant src, ASTNodeVariant dest, ASTNodeVariant m)
            : AbstractASTNode(loc), mSrc{std::move(src)}, mDest{std::move(dest)}, mMask{std::move(m)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string_impl() const { return "load"; };
        void print_impl(std::ostream &) const {};

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &src() const { return mSrc; }

        [[nodiscard]] const ASTNodeVariant &dest() const { return mDest; }

        [[nodiscard]] const ASTNodeVariant &mask() const { return mMask; }
    };
} // namespace voila::ast