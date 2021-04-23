#include "ast/Ge.hpp"

namespace voila::ast
{
    std::string Ge::type2string() const
    {
        return "ge";
    }
    bool Ge::is_ge() const
    {
        return true;
    }
    Ge *Ge::as_ge()
    {
        return this;
    }
    void Ge::print(std::ostream &ostream) const
    {
        ostream << ">";
    }
} // namespace voila::ast