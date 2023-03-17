#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/IExpression.hpp" // for IExpression
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class Scatter : public IExpression
    {
        ASTNodeVariant mIdxs;
        ASTNodeVariant mSrc;

      public:
        Scatter(Location loc, ASTNodeVariant idxs, ASTNodeVariant src_col);

        [[nodiscard]] bool is_scatter() const final;

        Scatter *as_scatter() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &idxs() const { return mIdxs; }

        [[nodiscard]] const ASTNodeVariant &src() const { return mSrc; }
    };

} // namespace voila::ast