#pragma once
#include "IExpression.hpp"     // for IExpression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{
    class Hash : public IExpression
    {
        std::vector<ASTNodeVariant> mItems;

      public:
        Hash(const Location loc, std::vector<ASTNodeVariant> items) : IExpression(loc), mItems{std::move(items)} {}

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_hash() const final;

        Hash *as_hash() final;

        void print(std::ostream &ostream) const override;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const std::vector<ASTNodeVariant> &items() const { return mItems; }
    };
} // namespace voila::ast