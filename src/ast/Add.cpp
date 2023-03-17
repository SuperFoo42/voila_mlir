#include "ast/Add.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast {
    bool Add::is_add() const {
        return true;
    }

    Add *Add::as_add() {
        return this;
    }

    std::string Add::type2string() const {
        return "add";
    }
    ASTNodeVariant Add::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Arithmetic::clone<Add>(vmap);
    }
} // namespace voila::ast