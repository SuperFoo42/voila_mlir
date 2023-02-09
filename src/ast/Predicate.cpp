#include "ast/Predicate.hpp"

#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/Expression.hpp"  // for Expression
#include "ast/IExpression.hpp" // for IExpression
#include <utility>

namespace voila::ast
{
    Predicate::Predicate(const Location loc, Expression expr) : IExpression(loc), mExpr(std::move(expr)) {}
    std::string Predicate::type2string() const { return "predicate"; }
    bool Predicate::is_predicate() const { return true; }
    Predicate *Predicate::as_predicate() { return this; }
    void Predicate::print(std::ostream &) const {}
    void Predicate::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Predicate::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Predicate::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        return std::make_shared<Predicate>(loc, mExpr.clone(vmap));
    }
} // namespace voila::ast
