#pragma once

#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"

#include <string>
#include <vector>

namespace voila::ast {
    class Lookup : public IExpression {
    public:
        Lookup(Location loc, std::vector<Expression> values, std::vector<Expression> tables, Expression hashes) :
                IExpression(loc), hashes{std::move(hashes)}, tables{std::move(tables)}, values{std::move(values)} {
            // TODO
        }

        [[nodiscard]] bool is_lookup() const final;

        Lookup *as_lookup() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        const Expression hashes;
        const std::vector<Expression> tables, values;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast