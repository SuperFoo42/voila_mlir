#pragma once
#include <utility>

#include "Expression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Read : public IExpression
    {
        Expression mColumn, mIdx;

      public:
        Read(Location loc, Expression lhs, Expression rhs) :
            IExpression(loc), mColumn{std::move(lhs)}, mIdx{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_read() const final;

        Read *as_read() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &column() const
        {
            return mColumn;
        }

        const Expression &idx() const
        {
            return mIdx;
        }
    };
} // namespace voila::ast