#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class FltConst : public AbstractASTNode<FltConst>
    {
      public:
        explicit FltConst(const Location loc, const double val) : AbstractASTNode<FltConst>(loc), val{val} {}

        [[nodiscard]] std::string type2string_impl() const { return "float"; }

        void print_impl(std::ostream &ostream) const { ostream << std::to_string(val); }

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &)
        {
            return std::make_shared<FltConst>(loc, val);
        }

        const double val;
    };
} // namespace voila::ast