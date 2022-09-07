#pragma once

#include "ASTVisitor.hpp"
#include "Comparison.hpp"
#include "IExpression.hpp"

#include <utility>
#include <vector>

namespace voila::ast {
    class Hash : public IExpression {
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

        const std::vector<Expression> &items() const {
            return mItems;
        }
    };
} // namespace voila::ast