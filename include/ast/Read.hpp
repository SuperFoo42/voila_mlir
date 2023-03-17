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
    class Read : public IExpression
    {
        ASTNodeVariant mColumn, mIdx;

      public:
        Read(Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs)
            : IExpression(loc), mColumn{std::move(lhs)}, mIdx{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_read() const final;

        Read *as_read() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &column() const { return mColumn; }

        [[nodiscard]] const ASTNodeVariant &idx() const { return mIdx; }
    };
} // namespace voila::ast