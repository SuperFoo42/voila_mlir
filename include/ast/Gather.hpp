#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/IExpression.hpp" // for IExpression
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Gather : public IExpression
    {
        ASTNodeVariant mColumn, mIdxs;

      public:
        Gather(const Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs)
            : IExpression(loc), mColumn{std::move(lhs)}, mIdxs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_gather() const final;

        Gather *as_gather() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &column() const { return mColumn; }

        [[nodiscard]] const ASTNodeVariant &idxs() const { return mIdxs; }
    };
} // namespace voila::ast