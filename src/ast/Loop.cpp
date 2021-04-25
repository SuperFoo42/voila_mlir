#include "ast/Loop.hpp"

namespace voila::ast
{
    std::string Loop::type2string() const
    {
        return "loop";
    }
    Loop *Loop::as_loop()
    {
        return this;
    }
    bool Loop::is_loop() const
    {
        return true;
    }
    void Loop::print(std::ostream &) const
    {
    }
    void Loop::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Loop::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast