#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <vector>               // for vector
#include "IStatement.hpp"       // for IStatement
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class FunctionCall : public IStatement
    {
        std::string mFun;
        std::vector<ASTNodeVariant> mArgs;

      public:
        FunctionCall(Location loc, std::string fun, std::vector<ASTNodeVariant> args);

        [[nodiscard]] bool is_function_call() const final;

        FunctionCall *as_function_call() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const std::string &fun() const {
            return mFun;
        }

        std::string &fun() {
            return mFun;
        }

        [[nodiscard]] const std::vector<ASTNodeVariant> &args() const {
            return mArgs;
        }
    };
} // namespace voila::ast