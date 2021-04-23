#include "ast/Aggregation.hpp"

namespace voila::ast
{
    bool Aggregation::is_aggr() const
    {
        return true;
    }
    Aggregation *Aggregation::as_aggr()
    {
        return this;
    }
    std::string Aggregation::type2string() const
    {
        return "aggregation";
    }
} // namespace voila::ast