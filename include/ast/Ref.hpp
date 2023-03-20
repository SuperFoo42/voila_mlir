#pragma once

#include <iosfwd> // for ostream
#include <memory> // for shared_ptr
#include <string> // for string

#include "ASTNodeVariant.hpp"
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap

namespace voila::ast
{
    class Ref : public AbstractASTNode<Ref>
    {
        ASTNodeVariant mRef;

      public:
        explicit Ref(Location loc, ASTNodeVariant ref) : AbstractASTNode<Ref>(loc), mRef{std::move(ref)} {}

        [[nodiscard]] std::string type2string_impl() const { return "reference"; }

        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &ref() const { return mRef; }
    };
} // namespace voila::ast