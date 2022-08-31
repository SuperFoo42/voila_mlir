#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"

#include <utility>

namespace voila::ast
{
    class Gather : public IExpression
    {
      public:
        Gather(const Location loc, Expression lhs, Expression rhs) :
            IExpression(loc), column{std::move(lhs)}, idxs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_gather() const final;

        Gather *as_gather() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression column, idxs;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast