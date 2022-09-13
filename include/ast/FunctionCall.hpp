#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class FunctionCall : public IStatement
    {
        std::string mFun;
        std::vector<Expression> mArgs;

      public:
        FunctionCall(Location loc, std::string fun, std::vector<Expression> args);

        [[nodiscard]] bool is_function_call() const final;

        FunctionCall *as_function_call() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        //void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const std::string &fun() const {
            return mFun;
        }

        std::string &fun() {
            return mFun;
        }

        const std::vector<Expression> &args() const {
            return mArgs;
        }
    };
} // namespace voila::ast