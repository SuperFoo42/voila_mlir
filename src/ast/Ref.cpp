#include "ast/Ref.hpp"

#include <utility>
#include "ast/Expression.hpp"
#include "ast/Variable.hpp"

namespace voila::ast {
    Ref::Ref(const Location loc, Expression var) : IExpression(loc), ref{std::move(var)} {
        // TODO find reference or error
    }

    bool Ref::is_reference() const {
        return true;
    }

    std::string Ref::type2string() const {
        return "reference";
    }

    const Ref *Ref::as_reference() const {
        return this;
    }

    void Ref::print(std::ostream &ostream) const {
        ostream << ref.as_variable()->var;
    }

    void Ref::visit(ASTVisitor &visitor) const {
        visitor(*this);
    }

    void Ref::visit(ASTVisitor &visitor) {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> Ref::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_unique<Ref>(loc, Expression::make(dynamic_cast<Variable*>(vmap.lookup(ref.as_variable()))->getptr()));
    }
} // namespace voila::ast