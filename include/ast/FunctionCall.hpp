#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <vector>               // for vector
#include "Expression.hpp"       // for Expression
#include "IStatement.hpp"       // for IStatement
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class ASTVisitor;

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

        [[nodiscard]] const std::string &fun() const {
            return mFun;
        }

        std::string &fun() {
            return mFun;
        }

        [[nodiscard]] const std::vector<Expression> &args() const {
            return mArgs;
        }
    };
} // namespace voila::ast