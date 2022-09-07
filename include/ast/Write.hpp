
#pragma once

#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast {
    class Write : public IStatement {
        Expression mDest;
        Expression mStart;
        Expression mSrc;

    public:
        Write(Location loc, Expression src_col, Expression dest_col, Expression wpos);

        [[nodiscard]] bool is_write() const final;

        Write *as_write() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &dest() const {
            return mDest;
        }

        const Expression &start() const {
            return mStart;
        }

        const Expression &src() const {
            return mSrc;
        }
    };

} // namespace voila::ast