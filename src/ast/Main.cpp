#include "ast/Main.hpp"
#include "ast/ASTNode.hpp"    // for Location
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include <utility>            // for move

namespace voila::ast
{
    Main::Main(const Location loc, std::vector<ASTNodeVariant> args, std::vector<ASTNodeVariant> exprs)
        : Fun(loc, "main", std::move(args), std::move(exprs))
    {
    }
    bool Main::is_main() const { return true; }
    Main *Main::as_main() { return this; }
    std::string Main::type2string() const { return "main function"; }
} // namespace voila::ast