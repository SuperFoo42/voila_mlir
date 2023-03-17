#include "ast/AggrSum.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/ASTNodeVariant.hpp"
#include "ASTNodes.hpp"

namespace voila::ast
{
    bool AggrSum::is_aggr_sum() const
    {
        return true;
    }
    std::string AggrSum::type2string() const
    {
        return "sum aggregation";
    }
    AggrSum *AggrSum::as_aggr_sum()
    {
        return this;
    }
    ASTNodeVariant AggrSum::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Aggregation::clone<AggrSum>(vmap);
    }
} // namespace voila::ast