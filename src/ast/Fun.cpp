#include "ast/Fun.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNode.hpp" // for ASTNode, Location
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"                    // for ASTVisitor
#include "ast/EmitNotLastStatementException.hpp" // for EmitNotLastStatemen...
#include "range/v3/range/concepts.hpp"
#include "range/v3/range/conversion.hpp"
#include "range/v3/range/concepts.hpp"
#include "range/v3/view/transform.hpp"
#include <algorithm>                             // for max, find_if
#include <ostream>                               // for operator<<, ostream
#include <utility>                               // for move

namespace voila::ast
{
    std::string Fun::type2string_impl() const { return "function definition"; }

    void Fun::print_impl(std::ostream &o) const
    {
        o << mName << "(";
        for (auto &arg : mArgs)
            std::visit(overloaded{[&o](auto &a) -> void { o << *a << ","; }, [](std::monostate) {}}, arg);
        o << ")";
    }

    ASTNodeVariant Fun::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        std::vector<ASTNodeVariant> clonedArgs;
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};;
        auto cvTransform =
            ranges::views::transform([&cloneVisitor](auto &arg) { return std::visit(cloneVisitor, arg); });

        return std::make_shared<Fun>(loc, mName, mArgs | cvTransform, mBody | cvTransform);
    }
} // namespace voila::ast