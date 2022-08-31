#include "ast/Fun.hpp"

#include "ast/ASTVisitor.hpp"

#include <utility>
#include "ast/EmitNotLastStatementException.hpp"
#include "range/v3/all.hpp"

namespace voila::ast
{
    Fun::Fun(const Location loc, std::string fun, std::vector<Expression> args, std::vector<Statement> exprs) :
        ASTNode(loc), name{std::move(fun)}, args{std::move(args)}, body{std::move(exprs)}
    {
        auto ret = std::find_if(body.begin(),body.end(), [](auto &e) -> auto {return e.is_emit();});
        if (ret == body.end())
        {
            result = std::nullopt;
        }
        else
        {
            if (ret != body.end()-1)
            {
                throw EmitNotLastStatementException();
            }
            result = *ret;
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
        o << name << "(";
        for (auto &arg : args)
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

    std::unique_ptr<ASTNode> Fun::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedArgs;
        ranges::transform(args, clonedArgs.begin(), [&vmap](auto &arg) { return arg.clone(vmap); });

        std::vector<Statement> clonedBody;
        ranges::transform(body, clonedBody.begin(), [&vmap](auto &stmt) { return stmt.clone(vmap); });


        return std::make_unique<Fun>(loc, name, clonedArgs, clonedBody);
    }
} // namespace voila::ast