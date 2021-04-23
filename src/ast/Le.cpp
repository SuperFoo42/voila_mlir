#include "ast/Le.hpp"

namespace voila::ast
{
    std::string Le::type2string() const
    {
        return "le";
    }
    bool Le::is_le() const
    {
        return true;
    }
    Le *Le::as_le()
    {
        return this;
    }
    void Le::print(std::ostream &ostream) const
    {
        ostream << "<";
    }
    void Le::checkArgs(Expression &lhs, Expression &rhs)
    {
        // TODO
    }
} // namespace voila::ast