#include "ast/Main.hpp"
#include "ast/ASTNode.hpp"    // for Location
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/Expression.hpp" // for Expression
#include "ast/Statement.hpp"  // for Statement
#include <utility>            // for move

namespace voila::ast
{
    Main::Main(const Location loc, std::vector<Expression> args, std::vector<Statement> exprs)
        : Fun(loc, "main", std::move(args), std::move(exprs))
    {
    }
    bool Main::is_main() const { return true; }
    Main *Main::as_main() { return this; }
    std::string Main::type2string() const { return "main function"; }

    void Main::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Main::visit(ASTVisitor &visitor) { visitor(*this); }
} // namespace voila::ast