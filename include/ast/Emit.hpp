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
    class Emit : public IStatement
    {
        std::vector<ASTNodeVariant> mExprs;

      public:
        explicit Emit(Location loc, std::vector<ASTNodeVariant> expr) : IStatement(loc), mExprs{std::move(expr)} {}

        [[nodiscard]] bool is_emit() const final;

        Emit *as_emit() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const std::vector<ASTNodeVariant> &exprs() const { return mExprs; }
    };

} // namespace voila::ast