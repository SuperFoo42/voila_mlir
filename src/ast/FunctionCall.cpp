#include "ast/FunctionCall.hpp"
#include <ostream>                           // for operator<<, ostream, bas...
#include <utility>                           // for move
#include "ast/ASTVisitor.hpp"                // for ASTVisitor
#include "ast/Expression.hpp"                // for Expression, operator<<
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

    FunctionCall::FunctionCall(const Location loc, std::string fun, std::vector<Expression> args) :
            IStatement(loc), mFun{std::move(fun)}, mArgs{std::move(args)} {
    }

    std::string FunctionCall::type2string() const {
        return "function call";
    }

    void FunctionCall::print(std::ostream &ostream) const {
        ostream << mFun << "(";
        for (const auto &arg: mArgs) {
            ostream << arg << ",";
        }
        ostream << ")";
    }
/*

    void FunctionCall::visit(ASTVisitor &visitor) const {
        visitor(*this);
    }
*/

    void FunctionCall::visit(ASTVisitor &visitor) {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> FunctionCall::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedArgs;
        ranges::transform(mArgs, clonedArgs.begin(), [&vmap](auto &arg) { return arg.clone(vmap); });
        return std::make_shared<FunctionCall>(loc, mFun, clonedArgs);
    }
} // namespace voila::ast