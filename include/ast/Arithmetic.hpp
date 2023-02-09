#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for make_shared, shared_ptr
#include <string>               // for string
#include "Expression.hpp"       // for Expression
#include "IExpression.hpp"      // for IExpression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class Arithmetic : public IExpression
    {
        Expression mLhs, mRhs;

      public:
        Arithmetic(Location loc, Expression lhs, Expression rhs);
        ~Arithmetic() override = default;

        [[nodiscard]] bool is_arithmetic() const final;

        Arithmetic *as_arithmetic() final;

        [[nodiscard]] std::string type2string() const override;
        void print(std::ostream &ostream) const final;

        [[nodiscard]] const Expression &lhs() const {
            return mLhs;
        }

        [[nodiscard]] const Expression &rhs() const {
            return mRhs;
        }
    protected:
        template <class T>
        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) requires std::is_base_of_v<Arithmetic, T> {
            return std::make_shared<T>(loc, mLhs.clone(vmap), mRhs.clone(vmap));
        }
    };
} // namespace voila::ast