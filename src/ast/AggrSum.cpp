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
    void AggrSum::print(std::ostream &ostream) const
    {
        ostream << "sum";
    }
} // namespace voila::ast