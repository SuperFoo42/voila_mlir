#include "ast/Sub.hpp"

namespace voila::ast
{
    std::string Sub::type2string() const
    {
        return "sub";
    }
    bool Sub::is_sub() const
    {
        return true;
    }
    void Sub::print(std::ostream &ostream) const
    {
        ostream << "-";
    }
} // namespace voila::ast