#include "ast/Add.hpp"

namespace voila::ast
{
    bool Add::is_add() const
    {
        return true;
    }
    Add *Add::as_add()
    {
        return this;
    }
    std::string Add::type2string() const
    {
        return "add";
    }
    void Add::print(std::ostream &ostream) const
    {
        ostream << "+";
    }
    void Add::checkArgs(Expression &lhs, Expression &rhs)
    { /*TODO*/
    }
} // namespace voila::ast