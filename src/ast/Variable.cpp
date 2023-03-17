#include "ast/Variable.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    bool Variable::is_variable() const { return true; }

    Variable *Variable::as_variable() { return this; }

    std::string Variable::type2string() const { return "variable"; }

    void Variable::print(std::ostream &ostream) const { ostream << var; }

    ASTNodeVariant Variable::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        if (vmap.count(this))
        {
            return dynamic_cast<Variable *>(vmap.lookup(this))->getptr();
        }

        std::shared_ptr<Variable> res = std::make_shared<Variable>(loc, var);
        vmap.insert(std::make_pair(this, res.get()));
        return res;
    }
} // namespace voila::ast
