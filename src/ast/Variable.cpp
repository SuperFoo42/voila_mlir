#include "ast/Variable.hpp"
namespace voila::ast
{
    bool Variable::is_variable() const
    {
        return true;
    }
    Variable *Variable::as_variable()
    {
        return this;
    }
    std::string Variable::type2string() const
    {
        return "variable";
    }
    void Variable::print(std::ostream &ostream) const
    {
        ostream << var;
    }
    void Variable::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Variable::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast
