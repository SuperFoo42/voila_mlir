#include "ast/AggrAvg.hpp"

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
    void AggrAvg::print(std::ostream &ostream) const
    {
        ostream << "average";
    }
} // namespace voila::ast