#pragma once
#include "Expression.hpp"      // for Expression
#include "IExpression.hpp"     // for IExpression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class ASTVisitor;

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

        Predicate *as_predicate() final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &expr() const { return mExpr; }
    };

} // namespace voila::ast
