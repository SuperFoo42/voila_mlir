#include "ast/AggrMin.hpp"

namespace voila::ast
{
    bool AggrMin::is_aggr_min() const
    {
        return true;
    }
    AggrMin *AggrMin::as_aggr_min()
    {
        return this;
    }
    std::string AggrMin::type2string() const
    {
        return "min aggregation";
    }
    void AggrMin::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void AggrMin::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast