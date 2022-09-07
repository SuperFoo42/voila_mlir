#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Selection : public IExpression
    {
        Expression mParam;
        Expression mPred;

      public:
        explicit Selection(const Location loc, Expression expr, Expression pred) : IExpression(loc), mParam(std::move(expr)), mPred(std::move(pred))
        {
            // TODO
        }

        [[nodiscard]] bool is_select() const final;

        Selection *as_select() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &param() const {
            return mParam;
        }

        const Expression &pred() const {
            return mPred;
        }
    };
} // namespace voila::ast