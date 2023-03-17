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
    class Selection : public IExpression
    {
        ASTNodeVariant mParam;
        ASTNodeVariant mPred;

      public:
        explicit Selection(const Location loc, ASTNodeVariant expr, ASTNodeVariant pred)
            : IExpression(loc), mParam(std::move(expr)), mPred(std::move(pred))
        {
            // TODO
        }

        [[nodiscard]] bool is_select() const final;

        Selection *as_select() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &param() const { return mParam; }

        [[nodiscard]] const ASTNodeVariant &pred() const { return mPred; }
    };
} // namespace voila::ast