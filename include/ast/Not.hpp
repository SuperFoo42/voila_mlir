#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Not : public AbstractASTNode<Not>
    {
        ASTNodeVariant mParam;

      public:
        explicit Not(const Location loc, ASTNodeVariant expr) : AbstractASTNode<Not>(loc), mParam(std::move(expr))
        {
            // TODO
        }

        void print_impl(std::ostream &) const {  };

        [[nodiscard]] std::string type2string_impl() const { return "not"; };

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &param() const { return mParam; }
    };

} // namespace voila::ast