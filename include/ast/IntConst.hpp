#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <cstdint>             // for intmax_t
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{

    class IntConst : public AbstractASTNode<IntConst>
    {

      public:
        explicit IntConst(Location loc, const std::intmax_t val) : AbstractASTNode<IntConst>(loc), val{val} {}

        [[nodiscard]] std::string type2string_impl() const { return "integer"; };

        void print_impl(std::ostream &ostream) const { ostream << std::to_string(val); };

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &)
        {
            return std::make_shared<IntConst>(loc, val);
        };

        const std::intmax_t val;
    };
} // namespace voila::ast