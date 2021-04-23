#include "ast/StrConst.hpp"

namespace voila::ast
{
    bool StrConst::is_string() const
    {
        return true;
    }
    std::string StrConst::type2string() const
    {
        return "string";
    }
    void StrConst::print(std::ostream &ostream) const
    {
        ostream << "\"" << val << "\"";
    }
} // namespace voila::ast