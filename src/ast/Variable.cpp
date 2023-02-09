#include "ast/Variable.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast {
    bool Variable::is_variable() const {
        return true;
    }

    Variable *Variable::as_variable() {
        return this;
    }

    std::string Variable::type2string() const {
        return "variable";
    }

    void Variable::print(std::ostream &ostream) const {
        ostream << var;
    }

    void Variable::visit(ASTVisitor &visitor) const {
        visitor(*this);
    }

    void Variable::visit(ASTVisitor &visitor) {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> Variable::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::shared_ptr<Variable> res = std::make_shared<Variable>(loc, var);
        vmap.insert(std::make_pair(this,res.get()));
        return res;
    }
} // namespace voila::ast
