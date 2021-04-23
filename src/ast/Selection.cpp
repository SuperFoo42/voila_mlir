#include "ast/Selection.hpp"

namespace voila::ast
{
    bool Selection::is_select() const
    {
        return true;
    }
    Selection *Selection::as_select()
    {
        return this;
    }
    std::string Selection::type2string() const
    {
        return "selection";
    }
    void Selection::checkArg(const Expression &)
    {
        // TODO: check param is boolean result
    }
} // namespace voila::ast