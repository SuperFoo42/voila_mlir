#pragma once

#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"

namespace voila::ast {
    class Ref : public IExpression {
        Expression mRef;

    public:
        explicit Ref(Location loc, Expression ref);

        [[nodiscard]] bool is_reference() const final;

        [[nodiscard]] std::string type2string() const override;

        [[nodiscard]] const Ref *as_reference() const final;

        void print(std::ostream &o) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &ref() const {
            return mRef;
        }
    };
} // namespace voila::ast