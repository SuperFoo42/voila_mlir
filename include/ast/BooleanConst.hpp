#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class BooleanConst : public AbstractASTNode<BooleanConst>
    {
      public:
        explicit BooleanConst(const Location loc, const bool val) : AbstractASTNode<BooleanConst>(loc), val{val} {}

        [[nodiscard]] std::string type2string_impl() const { return "bool"; }
        void print_impl(std::ostream &ostream) const { ostream << std::boolalpha << val << std::noboolalpha; };

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &)
        {
            return std::make_shared<BooleanConst>(loc, val);
        }

        const bool val;
    };
} // namespace voila::ast