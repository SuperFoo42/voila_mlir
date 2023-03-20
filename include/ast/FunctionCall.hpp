#pragma once
#include "ast/ASTNode.hpp" // for ASTNode (ptr only), Location
#include "range/v3/all.hpp"
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <vector>              // for vector

namespace voila::ast
{
    class FunctionCall : public AbstractASTNode<FunctionCall>
    {
        std::string mFun;
        std::vector<ASTNodeVariant> mArgs;

      public:
        FunctionCall(Location loc, std::string fun, ranges::input_range auto &&args)
            : AbstractASTNode<FunctionCall>(loc), mFun{std::move(fun)}, mArgs(ranges::to<std::vector>(args))
        {
        }

        [[nodiscard]] std::string type2string_impl() const { return "function call"; }

        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const std::string &fun() const { return mFun; }

        std::string &fun() { return mFun; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &args() const { return mArgs; }
    };
} // namespace voila::ast