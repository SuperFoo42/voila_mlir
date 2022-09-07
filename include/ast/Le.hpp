#pragma once
#include "Comparison.hpp"
#include "Expression.hpp"
#include "ASTVisitor.hpp"

namespace voila::ast
{
    class Le : public Comparison
    {
      public:
        Le(const Location loc, Expression lhs, Expression rhs) : Comparison(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_le() const final;

        Le *as_le() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Comparison::clone<Le>(vmap);
        }
    };
} // namespace voila::ast