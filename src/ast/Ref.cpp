#include "ast/Ref.hpp"

namespace voila::ast
{
    Ref::Ref(const std::string &var) : IExpression()
    {
        // TODO find reference or error
    }
    bool Ref::is_reference() const
    {
        return true;
    }
    std::string Ref::type2string() const
    {
        return "reference";
    }
    Ref *Ref::as_reference()
    {
        return this;
    }
    void Ref::print(std::ostream &o) const
    {
        o << "Reference to "; // TODO << ref.type2string();
    }
} // namespace voila::ast