#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class Write : public AbstractASTNode<Write>
    {
        ASTNodeVariant mDest;
        ASTNodeVariant mStart;
        ASTNodeVariant mSrc;

      public:
        Write(Location loc, ASTNodeVariant src_col, ASTNodeVariant dest_col, ASTNodeVariant wpos)
            : AbstractASTNode<Write>(loc), mDest{std::move(dest_col)}, mStart{std::move(wpos)}, mSrc{std::move(src_col)}
        {
        }

        [[nodiscard]] std::string type2string_impl() const { return "write"; }

        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &dest() const { return mDest; }
        [[nodiscard]] const ASTNodeVariant &start() const { return mStart; }
        [[nodiscard]] const ASTNodeVariant &src() const { return mSrc; }
    };
} // namespace voila::ast