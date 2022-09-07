#pragma once
#include "IExpression.hpp"
#include "ASTVisitor.hpp"
#include <utility>

namespace voila::ast
{
    class Variable : public IExpression, virtual public std::enable_shared_from_this<Variable>
    {
      public:
        //TODO: private ctor to mitigate stack allocations and resulting getptr nullptr
        explicit Variable(const Location loc, std::string val) : IExpression(loc), var{std::move(val)} {}

        [[nodiscard]] bool is_variable() const final;

        Variable *as_variable() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<Variable> getptr() {
            return shared_from_this();
        }

        const std::string var;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast