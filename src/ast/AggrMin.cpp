#include "ast/AggrMin.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/ASTNodeVariant.hpp"
#include "ASTNodes.hpp"

namespace voila::ast
{
    bool AggrMin::is_aggr_min() const { return true; }
    AggrMin *AggrMin::as_aggr_min() { return this; }
    std::string AggrMin::type2string() const { return "min aggregation"; }
    ASTNodeVariant AggrMin::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Aggregation::clone<AggrMin>(vmap);
    }
} // namespace voila::ast