#include "ast/Logical.hpp"

namespace voila::ast
{
    bool Logical::is_logical() const
    {
        return true;
    }
    Logical *Logical::as_logical()
    {
        return this;
    }
    std::string Logical::type2string() const
    {
        return "logical";
    }
} // namespace voila::ast