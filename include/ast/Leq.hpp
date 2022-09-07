#pragma once

#include "ASTVisitor.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Leq : public Comparison
    {
      public:
        Leq(const Location loc, Expression lhs, Expression rhs) : Comparison(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_leq() const final;

        Leq *as_leq() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Comparison::clone<Leq>(vmap);
        }
    };
} // namespace voila::ast