#pragma once
#include "Expression.hpp"
#include "IExpression.hpp"

#include <utility>
#include <vector>
#include "ASTVisitor.hpp"
namespace voila::ast
{
    //TODO
    /**
     * @deprecated ?
     */
    class TupleCreate : public IExpression
    {
      public:
        explicit TupleCreate(Location loc,std::vector<Expression> tupleElems);

        [[nodiscard]] bool is_tuple_create() const final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        const std::vector<Expression> elems;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast