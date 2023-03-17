#pragma once
#include "IStatement.hpp"      // for IStatement
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{

    // TODO: fix this
    class Loop : public IStatement
    {
        ASTNodeVariant mPred;
        std::vector<ASTNodeVariant> mStms;

      public:
        Loop(const Location loc, ASTNodeVariant pred, std::vector<ASTNodeVariant> stms)
            : IStatement(loc), mPred{std::move(pred)}, mStms{std::move(stms)}
        {
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_loop() const final;

        Loop *as_loop() final;
        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;
        // TODO
        // CrossingVariables crossing_variables;

        [[nodiscard]] const ASTNodeVariant &pred() const { return mPred; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &stmts() const { return mStms; }
    };

} // namespace voila::ast