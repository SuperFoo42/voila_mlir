#include "ast/AggrAvg.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/ASTNodeVariant.hpp"
#include "ASTNodes.hpp"

namespace voila::ast
{
    bool AggrAvg::is_aggr_avg() const
    {
        return true;
    }
    std::string AggrAvg::type2string() const
    {
        return "avg aggregation";
    }
    AggrAvg *AggrAvg::as_aggr_avg()
    {
        return this;
    }
    ASTNodeVariant AggrAvg::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Aggregation::clone<AggrAvg>(vmap);
    }
} // namespace voila::ast