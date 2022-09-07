#pragma once

#include "Expression.hpp"
#include "IStatement.hpp"
#include "Statement.hpp"
#include "ASTVisitor.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace voila::ast {
    /**
     * @brief Meta node to wrap expressions into statements
     *
     */
    class StatementWrapper : public IStatement {
        Expression mExpr;

    public:
        explicit StatementWrapper(Location loc, Expression expr);

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_statement_wrapper() const final;

        [[nodiscard]] StatementWrapper *as_statement_wrapper() final;

        std::optional<Expression> as_expression() final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &expr() const
        {
            return mExpr;
        }
    };

} // namespace voila::ast