#pragma once
#include "Arithmetic.hpp"
#include "Expression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Mod : public Arithmetic
    {
      public:
        Mod(Location loc, Expression lhs, Expression rhs) : Arithmetic(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_mod() const final;

        Mod *as_mod() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Arithmetic::clone<Mod>(vmap);
        }
    };
} // namespace voila::ast