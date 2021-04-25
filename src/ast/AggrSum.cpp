#include "ast/AggrSum.hpp"

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
    void AggrSum::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void AggrSum::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast