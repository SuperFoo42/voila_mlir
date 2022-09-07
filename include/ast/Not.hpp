#pragma once
#include <utility>

#include "Expression.hpp"
#include "Logical.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Not : public Logical
    {
        Expression mParam;

      public:
        explicit Not(const Location loc,Expression expr) :
            Logical(loc), mParam(std::move(expr))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_not() const final;

        Not *as_not() final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &param() const {
            return mParam;
        }
    };
} // namespace voila::ast