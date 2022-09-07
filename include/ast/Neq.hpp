#pragma once
#include "ASTVisitor.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Neq : public Comparison
    {
      public:
        Neq(const Location loc, Expression lhs, Expression rhs) : Comparison(loc, std::move(lhs), std::move(rhs))
        { // TODO
        }
        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_neq() const final;

        Neq *as_neq() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Comparison::clone<Neq>(vmap);
        }
    };
} // namespace voila::ast