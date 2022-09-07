#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"

#include <utility>

namespace voila::ast
{
    class Gather : public IExpression
    {
        Expression mColumn, mIdxs;

      public:
        Gather(const Location loc, Expression lhs, Expression rhs) :
            IExpression(loc), mColumn{std::move(lhs)}, mIdxs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_gather() const final;

        Gather *as_gather() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &column() const {
            return mColumn;
        }

        const Expression &idxs() const {
            return mIdxs;
        }
    };
} // namespace voila::ast