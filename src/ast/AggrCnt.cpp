#include "ast/AggrCnt.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    bool AggrCnt::is_aggr_cnt() const { return true; }
    AggrCnt *AggrCnt::as_aggr_cnt() { return this; }
    std::string AggrCnt::type2string() const { return "count aggregation"; }
    ASTNodeVariant AggrCnt::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Aggregation::clone<AggrCnt>(vmap);
    }
} // namespace voila::ast