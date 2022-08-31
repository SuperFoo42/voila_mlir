#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"

#include <string>
#include <vector>
#include <utility>

namespace voila::ast
{
    class Insert : public IExpression
    {
      public:
        Insert(Location loc, Expression keys, std::vector<Expression> values) :
            IExpression(loc), keys{std::move(keys)}, values{std::move(values)}
        {
            // TODO
        }

        [[nodiscard]] bool is_insert() const final;

        Insert *as_insert() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        const Expression keys;
        const std::vector<Expression> values;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast