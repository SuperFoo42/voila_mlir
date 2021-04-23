#include "ast/Const.hpp"

namespace voila::ast
{
    bool Const::is_const() const
    {
        return true;
    }
    Const *Const::as_const()
    {
        return this;
    }
    std::string Const::type2string() const
    {
        return "constant";
    }
} // namespace voila::ast