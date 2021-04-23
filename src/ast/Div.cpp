#include "ast/Div.hpp"

namespace voila::ast
{
    std::string Div::type2string() const
    {
        return "div";
    }
    bool Div::is_div() const
    {
        return true;
    }
    Div *Div::as_div()
    {
        return this;
    }
    void Div::print(std::ostream &ostream) const
    {
        ostream << "/";
    }
} // namespace voila::ast