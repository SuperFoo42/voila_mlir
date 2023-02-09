#include "ast/Emit.hpp"
#include <algorithm>           // for max
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/Expression.hpp"  // for Expression

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

    std::shared_ptr<ASTNode> Emit::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedExprs;
        for (auto &arg : mExprs){
            auto tmp =  arg.clone(vmap);
            clonedExprs.push_back(tmp);
        }

        return std::make_shared<Emit>(loc, clonedExprs);
    }
} // namespace voila::ast