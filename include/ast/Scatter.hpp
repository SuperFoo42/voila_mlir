#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>
#include "ASTVisitor.hpp"

namespace voila::ast
{
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

        const Expression &idxs() const
        {
            return mIdxs;
        }

        const Expression &src() const {
            return mSrc;
        }
    };

} // namespace voila::ast