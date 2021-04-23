#include "ast/IntConst.hpp"

namespace voila::ast
{
    bool IntConst::is_integer() const
    {
        return true;
    }
    IntConst *IntConst::as_integer()
    {
        return this;
    }
    std::string IntConst::type2string() const
    {
        return "integer";
    }
    void IntConst::print(std::ostream &ostream) const
    {
        ostream << std::to_string(val);
    }
} // namespace voila::ast