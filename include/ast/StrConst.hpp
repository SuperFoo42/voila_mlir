#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class StrConst : public AbstractASTNode<StrConst>
    {
      public:
        explicit StrConst(const Location loc, std::string val) : AbstractASTNode<StrConst>(loc), val{std::move(val)} {}

        [[nodiscard]] std::string type2string_impl() const { return "string"; }

        void print_impl(std::ostream &ostream) const { ostream << "\"" << val << "\""; }

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &)
        {
            return std::make_shared<StrConst>(loc, val);
        };

        const std::string val;
    };
} // namespace voila::ast