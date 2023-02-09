#pragma once
#include "Expression.hpp"      // for Expression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/IExpression.hpp" // for IExpression
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class ASTVisitor;

    class Scatter : public IExpression
    {
        Expression mIdxs;
        Expression mSrc;

      public:
        Scatter(Location loc, Expression idxs, Expression src_col);

        [[nodiscard]] bool is_scatter() const final;

        Scatter *as_scatter() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &idxs() const { return mIdxs; }

        const Expression &src() const { return mSrc; }
    };

} // namespace voila::ast