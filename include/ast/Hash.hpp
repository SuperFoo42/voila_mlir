#pragma once
#include "IExpression.hpp"     // for IExpression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/Expression.hpp"  // for Expression
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{
    class ASTVisitor;

    class Hash : public IExpression
    {
        std::vector<Expression> mItems;

      public:
        Hash(const Location loc, std::vector<Expression> items) : IExpression(loc), mItems{std::move(items)} {}

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_hash() const final;

        Hash *as_hash() final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        void print(std::ostream &ostream) const override;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const std::vector<Expression> &items() const { return mItems; }
    };
} // namespace voila::ast