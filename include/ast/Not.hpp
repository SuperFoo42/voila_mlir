#pragma once
#include "Logical.hpp"         // for Logical
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Not : public Logical
    {
        ASTNodeVariant mParam;

      public:
        explicit Not(const Location loc, ASTNodeVariant expr) : Logical(loc), mParam(std::move(expr))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_not() const final;

        Not *as_not() final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &param() const { return mParam; }
    };
} // namespace voila::ast