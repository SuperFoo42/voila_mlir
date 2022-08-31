#include "ast/Emit.hpp"
#include "range/v3/all.hpp"
namespace voila::ast
{
    bool Emit::is_emit() const
    {
        return true;
    }

    Emit *Emit::as_emit()
    {
        return this;
    }

    std::string Emit::type2string() const
    {
        return "emit";
    }

    void Emit::print(std::ostream &) const
    {
    }
    void Emit::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Emit::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> Emit::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedExprs;
        ranges::transform(exprs, clonedExprs.begin(), [&vmap](auto ex) {return ex.clone(vmap); });

        return std::make_unique<Emit>(loc, clonedExprs);
    }
} // namespace voila::ast