#include <utility>

#include "ast/Predicate.hpp"

namespace voila::ast
{
    Predicate::Predicate(const Location loc, Expression expr) :
        IExpression(loc), expr(std::move(expr)){
    }
    std::string Predicate::type2string() const
    {
        return "predicate";
    }
    bool Predicate::is_predicate() const
    {
        return true;
    }
    Predicate *Predicate::as_predicate()
    {
        return this;
    }
    void Predicate::print(std::ostream &) const {}
    void Predicate::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Predicate::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> Predicate::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_unique<Predicate>(loc, expr.clone(vmap));
    }
}
