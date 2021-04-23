#include "ast/Comparison.hpp"

namespace voila::ast
{
    bool Comparison::is_comparison() const
    {
        return true;
    }
    Comparison *Comparison::as_comparison()
    {
        return this;
    }
    std::string Comparison::type2string() const
    {
        return "comparison";
    }
} // namespace voila::ast