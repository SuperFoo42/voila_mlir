#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <vector>               // for vector
#include "Expression.hpp"       // for Expression
#include "IExpression.hpp"      // for IExpression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{class ASTVisitor;

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

        std::vector<Expression> elems;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast