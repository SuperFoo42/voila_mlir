#include "ast/Ref.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/Expression.hpp"
#include "ast/IExpression.hpp" // for IExpression
#include "ast/Variable.hpp"
#include <utility>

namespace voila::ast
{
    Ref::Ref(const Location loc, Expression var) : IExpression(loc), mRef{std::move(var)}
    {
        // TODO find reference or error
    }

    bool Ref::is_reference() const { return true; }

    std::string Ref::type2string() const { return "reference"; }

    const Ref *Ref::as_reference() const { return this; }

    void Ref::print(std::ostream &ostream) const { ostream << mRef.as_variable()->var; }

    void Ref::visit(ASTVisitor &visitor) const { visitor(*this); }

    void Ref::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Ref::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        auto ptr = dynamic_cast<Variable *>(vmap.lookup(mRef.as_variable()))->getptr();
        return std::make_shared<Ref>(loc, Expression::make(ptr));
    }
} // namespace voila::ast