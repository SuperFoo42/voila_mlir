#include "ast/Not.hpp"

namespace voila::ast
{
    std::string Not::type2string() const
    {
        return "not";
    }
    bool Not::is_not() const
    {
        return true;
    }
    Not *Not::as_not()
    {
        return this;
    }
    void Not::print(std::ostream &ostream) const
    {
        ostream << "!";
    }
} // namespace voila::ast