#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Selection : public IExpression
    {
      public:
        const Expression param;
        const Expression pred;
        explicit Selection(const Location loc, Expression expr, Expression pred) : IExpression(loc), param(std::move(expr)), pred(std::move(pred))
        {
            // TODO
        }

        [[nodiscard]] bool is_select() const final;

        Selection *as_select() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast