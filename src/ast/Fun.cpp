#include "ast/Fun.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNode.hpp" // for ASTNode, Location
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"                    // for ASTVisitor
#include "ast/EmitNotLastStatementException.hpp" // for EmitNotLastStatemen...
#include <algorithm>                             // for max, find_if
#include <ostream>                               // for operator<<, ostream
#include <utility>                               // for move

namespace voila::ast
{
    Fun::Fun(const Location loc, std::string fun, std::vector<ASTNodeVariant> args, std::vector<ASTNodeVariant> exprs)
        : AbstractASTNode(loc), mName{std::move(fun)}, mArgs{std::move(args)}, mBody{std::move(exprs)}, mResult{std::monostate()}
    {
        auto ret = std::find_if(mBody.begin(), mBody.end(),
                                [](auto &e) -> auto { return std::holds_alternative<std::shared_ptr<Emit>>(e); });
        if (ret != mBody.end())
        {
            if (ret != mBody.end() - 1)
            {
                throw EmitNotLastStatementException();
            }
            mResult = *ret;
        }
    }
    bool Fun::is_function_definition() const { return true; }
    Fun *Fun::as_function_definition() { return this; }
    std::string Fun::type2string() const { return "function definition"; }
    void Fun::print(std::ostream &o) const
    {
        o << mName << "(";
        for (auto &arg : mArgs)
            std::visit(overloaded{[&o](auto &a) -> void { o << *a << ","; }, [](std::monostate) {}}, arg);
        o << ")";
    }

    ASTNodeVariant Fun::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        std::vector<ASTNodeVariant> clonedArgs;
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        for (auto &arg : mArgs)
        {
            auto tmp = std::visit(cloneVisitor, arg);
            clonedArgs.push_back(tmp);
        }

        std::vector<ASTNodeVariant> clonedBody;
        for (auto &arg : mBody)
        {
            auto tmp = std::visit(cloneVisitor, arg);
            clonedBody.push_back(tmp);
        }

        return std::make_shared<Fun>(loc, mName, clonedArgs, clonedBody);
    }
} // namespace voila::ast