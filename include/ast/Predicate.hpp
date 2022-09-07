#pragma once
#include "IExpression.hpp"
#include "ASTVisitor.hpp"
#include "Expression.hpp"

namespace voila::ast
{

    /**
     * @brief Meta node to wrap expressions into predicates
     * @deprecated
     */
    class Predicate : public IExpression
    {
        Expression mExpr;

      public:
        explicit Predicate(Location loc, Expression expr);

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_predicate() const final;

        Predicate * as_predicate() final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &expr() const {
            return mExpr;
        }
    };

} // namespace voila::ast
