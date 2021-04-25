#include "ast/AggrMax.hpp"

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
    void AggrMax::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void AggrMax::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast