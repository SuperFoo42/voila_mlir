#include "ast/FunctionCall.hpp"
#include "range/v3/all.hpp"

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