#include "ast/AggrMax.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/ASTNodeVariant.hpp"
#include "ASTNodes.hpp"

namespace voila::ast
{
    bool AggrMax::is_aggr_max() const
    {
        return true;
    }
    AggrMax *AggrMax::as_aggr_max()
    {
        return this;
    }
    std::string AggrMax::type2string() const
    {
        return "max aggregation";
    }
    ASTNodeVariant AggrMax::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Aggregation::clone<AggrMax>(vmap);
    }
} // namespace voila::ast