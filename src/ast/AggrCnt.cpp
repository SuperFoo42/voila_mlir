#include "ast/AggrCnt.hpp"

namespace voila::ast
{
    bool AggrCnt::is_aggr_cnt() const
    {
        return true;
    }
    AggrCnt *AggrCnt::as_aggr_cnt()
    {
        return this;
    }
    std::string AggrCnt::type2string() const
    {
        return "count aggregation";
    }
    void AggrCnt::print(std::ostream &ostream) const
    {
        ostream << "count";
    }
} // namespace voila::ast