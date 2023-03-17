#include "ast/Read.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    bool Read::is_read() const { return true; }
    Read *Read::as_read() { return this; }
    std::string Read::type2string() const { return "read"; }
    void Read::print(std::ostream &) const {}

    ASTNodeVariant Read::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Read>(loc, std::visit(cloneVisitor, mColumn), std::visit(cloneVisitor, mIdx));
    }
} // namespace voila::ast