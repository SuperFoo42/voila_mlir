#include "ast/Predicate.hpp"

#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/IExpression.hpp" // for IExpression
#include <utility>

namespace voila::ast
{
    Predicate::Predicate(const Location loc, ASTNodeVariant expr) : IExpression(loc), mExpr(std::move(expr)) {}
    std::string Predicate::type2string() const { return "predicate"; }
    bool Predicate::is_predicate() const { return true; }
    Predicate *Predicate::as_predicate() { return this; }
    void Predicate::print(std::ostream &) const {}
    ASTNodeVariant Predicate::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Predicate>(loc, std::visit(cloneVisitor, mExpr));
    }
} // namespace voila::ast
