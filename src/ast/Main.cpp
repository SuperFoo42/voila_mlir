#include "ast/Main.hpp"

namespace voila::ast
{
    Main::Main(std::vector<std::string> args, std::vector<Statement> exprs) :
        Fun("main", std::move(args), std::move(exprs))
    {
        // TODO register as entry point and check args + exprs
    }
    bool Main::is_main() const
    {
        return true;
    }
    Main *Main::as_main()
    {
        return this;
    }
    std::string Main::type2string() const
    {
        return "main function";
    }

    void Main::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Main::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast