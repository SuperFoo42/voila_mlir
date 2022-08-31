#include "ast/Comparison.hpp"

namespace voila::ast
{
    bool Comparison::is_comparison() const
    {
        return true;
    }
    Comparison *Comparison::as_comparison()
    {
        return this;
    }
    std::string Comparison::type2string() const
    {
        return "comparison";
    }
    void Comparison::print(std::ostream &) const {}

    std::unique_ptr<ASTNode> Comparison::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_unique<Comparison>(loc, lhs.clone(vmap), rhs.clone(vmap));
    }
} // namespace voila::ast