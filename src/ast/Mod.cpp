#include "ast/Mod.hpp"

namespace voila::ast
{
    std::string Mod::type2string() const
    {
        return "mod";
    }
    bool Mod::is_mod() const
    {
        return true;
    }
    Mod *Mod::as_mod()
    {
        return this;
    }
    void Mod::print(std::ostream &ostream) const
    {
        ostream << "%";
    }
} // namespace voila::ast