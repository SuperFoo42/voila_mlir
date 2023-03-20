#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class Scatter : public AbstractASTNode<Scatter>
    {
        ASTNodeVariant mIdxs;
        ASTNodeVariant mSrc;

      public:
        Scatter(Location loc, ASTNodeVariant idxs, ASTNodeVariant src_col)
            : AbstractASTNode<Scatter>(loc), mIdxs{std::move(idxs)}, mSrc{std::move(src_col)}
        {
        }

        [[nodiscard]] std::string type2string_impl() const { return "scatter"; };
        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &idxs() const { return mIdxs; }

        [[nodiscard]] const ASTNodeVariant &src() const { return mSrc; }
    };

} // namespace voila::ast