#include "ast/BooleanConst.hpp"

namespace voila::ast
{
    bool BooleanConst::is_bool() const
    {
        return true;
    }
    BooleanConst *BooleanConst::as_bool()
    {
        return this;
    }
    std::string BooleanConst::type2string() const
    {
        return "bool";
    }
    void BooleanConst::print(std::ostream &ostream) const
    {
        ostream << std::to_string(val);
    }
} // namespace voila::ast