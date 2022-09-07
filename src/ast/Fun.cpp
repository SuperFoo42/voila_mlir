#include "ast/Fun.hpp"

#include "ast/ASTVisitor.hpp"

#include <utility>
#include "ast/EmitNotLastStatementException.hpp"
#include <algorithm>

namespace voila::ast
{
    Fun::Fun(const Location loc, std::string fun, std::vector<Expression> args, std::vector<Statement> exprs) :
        ASTNode(loc), mName{std::move(fun)}, mArgs{std::move(args)}, mBody{std::move(exprs)}
    {
        auto ret = std::find_if(mBody.begin(),mBody.end(), [](auto &e) -> auto {return e.is_emit();});
        if (ret == mBody.end())
        {
            mResult = std::nullopt;
        }
        else
        {
            if (ret != mBody.end()-1)
            {
                throw EmitNotLastStatementException();
            }
            mResult = *ret;
        }
    }
    bool Fun::is_function_definition() const
    {
        return true;
    }
    Fun *Fun::as_function_definition()
    {
        return this;
    }
    std::string Fun::type2string() const
    {
        return "function definition";
    }
    void Fun::print(std::ostream &o) const
    {
        o << mName << "(";
        for (auto &arg : mArgs)
            o << arg << ",";
        o << ")";
    }
    void Fun::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Fun::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> Fun::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedArgs;

        for (auto &arg : mArgs){
            auto tmp =  arg.clone(vmap);
            clonedArgs.push_back(tmp);
        }

        std::vector<Statement> clonedBody;
        for (auto &arg : mBody){
            auto tmp =  arg.clone(vmap);
            clonedBody.push_back(tmp);
        }


        return std::make_shared<Fun>(loc, mName, clonedArgs, clonedBody);
    }
} // namespace voila::ast