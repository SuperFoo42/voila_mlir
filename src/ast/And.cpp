#include "ast/And.hpp"

namespace voila::ast
{
    std::string And::type2string() const
    {
        return "and";
    }
    bool And::is_and() const
    {
        return true;
    }
    And *And::as_and()
    {
        return this;
    }
    void And::print(std::ostream &ostream) const
    {
        ostream << "&&";
    }
    void And::checkArgs(Expression &lhs, Expression &rhs)
    {
        // TODO
    }
} // namespace voila::ast