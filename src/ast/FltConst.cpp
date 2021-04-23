#include "ast/FltConst.hpp"

namespace voila::ast
{
    bool FltConst::is_float() const
    {
        return true;
    }
    FltConst *FltConst::as_float()
    {
        return this;
    }
    std::string FltConst::type2string() const
    {
        return "float";
    }
    void FltConst::print(std::ostream &ostream) const
    {
        ostream << std::to_string(val);
    }
} // namespace voila::ast