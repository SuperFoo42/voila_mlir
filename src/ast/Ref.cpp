#include "ast/Ref.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/IExpression.hpp" // for IExpression
#include "ast/Variable.hpp"
#include <utility>

namespace voila::ast
{
    Ref::Ref(const Location loc, ASTNodeVariant var) : IExpression(loc), mRef{std::move(var)}
    {
        // TODO find reference or error
    }

    bool Ref::is_reference() const { return true; }

    std::string Ref::type2string() const { return "reference"; }

    const Ref *Ref::as_reference() const { return this; }

    void Ref::print(std::ostream &ostream) const
    {
        std::visit(overloaded{[&ostream](std::shared_ptr<Variable> &var) { ostream << var->var; }, [](auto) {}}, mRef);
    }

    ASTNodeVariant Ref::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Ref>(loc, std::visit(cloneVisitor, mRef));
    }
} // namespace voila::ast