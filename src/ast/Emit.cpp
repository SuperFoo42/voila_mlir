#include "ast/Emit.hpp"

namespace voila::ast
{
    bool Emit::is_emit() const
    {
        return true;
    }
    Emit *Emit::as_emit()
    {
        return this;
    }
    std::string Emit::type2string() const
    {
        return "emit";
    }
    void Emit::print(std::ostream &ostream) const
    {
        ostream << "emit";
    }
} // namespace voila::ast