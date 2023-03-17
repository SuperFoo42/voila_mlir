#include "ast/Leq.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast {
    std::string Leq::type2string() const {
        return "leq";
    }

    bool Leq::is_leq() const {
        return true;
    }

    Leq *Leq::as_leq() {
        return this;
    }
    ASTNodeVariant Leq::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Comparison::clone<Leq>(vmap);
    }
} // namespace voila::ast