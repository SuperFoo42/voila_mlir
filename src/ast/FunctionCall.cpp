#include "ast/FunctionCall.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/functional/identity.hpp" // for identity
#include "range/v3/view/transform.hpp"      // for transform, transform_fn
#include <ostream>                          // for operator<<, ostream, bas...
#include <utility>                          // for move

namespace voila::ast
{

    void FunctionCall::print_impl(std::ostream &ostream) const
    {
        ostream << mFun << "(";
        for (const auto &arg : mArgs)
        {
            std::visit(overloaded{[&ostream](auto &a) -> void { ostream << *a << ","; }, [](std::monostate) {}}, arg);
        }
        ostream << ")";
    }

    ASTNodeVariant FunctionCall::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor =  overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};;
        return std::make_shared<FunctionCall>(
            loc, mFun,
            mArgs | ranges::views::transform([&cloneVisitor](auto &arg) { return std::visit(cloneVisitor, arg); }));
    }
} // namespace voila::ast