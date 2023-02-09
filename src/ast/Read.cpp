#include "ast/Read.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/Expression.hpp" // for Expression

namespace voila::ast
{
    bool Read::is_read() const { return true; }
    Read *Read::as_read() { return this; }
    std::string Read::type2string() const { return "read"; }
    void Read::print(std::ostream &) const {}
    void Read::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Read::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Read::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        return std::make_shared<Read>(loc, mColumn.clone(vmap), mIdx.clone(vmap));
    }
} // namespace voila::ast