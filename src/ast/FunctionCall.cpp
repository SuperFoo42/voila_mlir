#include "ast/FunctionCall.hpp"
#include <ostream>                           // for operator<<, ostream, bas...
#include <utility>                           // for move
#include "ast/ASTVisitor.hpp"                // for ASTVisitor
#include "ast/ASTNodeVariant.hpp"
#include "ASTNodes.hpp"
#include "ast/IStatement.hpp"                // for IStatement
#include "range/v3/algorithm/transform.hpp"  // for transform, transform_fn
#include "range/v3/functional/identity.hpp"  // for identity

namespace voila::ast {
    bool FunctionCall::is_function_call() const {
        return true;
    }

    FunctionCall *FunctionCall::as_function_call() {
        return this;
    }

    FunctionCall::FunctionCall(const Location loc, std::string fun, std::vector<ASTNodeVariant> args) :
            IStatement(loc), mFun{std::move(fun)}, mArgs{std::move(args)} {
    }

    std::string FunctionCall::type2string() const {
        return "function call";
    }

    void FunctionCall::print(std::ostream &ostream) const {
        ostream << mFun << "(";
        for (const auto &arg: mArgs) {
            std::visit(overloaded{[&ostream](auto &a) {ostream << *a << ",";}, [](std::monostate ) {}}, arg);
        }
        ostream << ")";
    }

    ASTNodeVariant FunctionCall::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) {
        std::vector<ASTNodeVariant> clonedArgs;
        auto cloneVisitor =
            overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        ranges::transform(mArgs, clonedArgs.begin(), [&cloneVisitor](auto &arg) { return std::visit(cloneVisitor,arg);});
        return std::make_shared<FunctionCall>(loc, mFun, clonedArgs);
    }
} // namespace voila::ast