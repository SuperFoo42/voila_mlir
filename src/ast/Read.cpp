#include "ast/Read.hpp"

namespace voila::ast
{
    bool Read::is_read() const
    {
        return true;
    }
    Read *Read::as_read()
    {
        return this;
    }
    std::string Read::type2string() const
    {
        return "read";
    }
    void Read::checkArgs(Expression &lhs, Expression &rhs)
    {
        // TODO
    }
} // namespace voila::ast