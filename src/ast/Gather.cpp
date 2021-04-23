#include "ast/Gather.hpp"

namespace voila::ast
{
    bool Gather::is_gather() const
    {
        return true;
    }
    Gather *Gather::as_gather()
    {
        return this;
    }
    std::string Gather::type2string() const
    {
        return "gather";
    }
} // namespace voila::ast