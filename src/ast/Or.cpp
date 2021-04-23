#include "ast/Or.hpp"

namespace voila::ast
{
    std::string Or::type2string() const
    {
        return "or";
    }
    bool Or::is_or() const
    {
        return true;
    }
    Or *Or::as_or()
    {
        return this;
    }
    void Or::print(std::ostream &ostream) const
    {
        ostream << "||";
    }
} // namespace voila::ast