#pragma once
#include "IStatement.hpp"      // for IStatement
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <optional>            // for optional
#include <string>              // for string
#include <vector>              // for vector

namespace voila::ast
{
    class Assign : public IStatement
    {
        ASTNodeVariant pred;
        std::vector<ASTNodeVariant> mDdests;
        ASTNodeVariant mExpr;

      public:
        Assign(Location loc, std::vector<ASTNodeVariant> dests, ASTNodeVariant expr);

        [[nodiscard]] bool is_assignment() const final;

        Assign *as_assignment() final;

        [[nodiscard]] std::string type2string() const final;

        void set_predicate(ASTNodeVariant expression) final;
        std::optional<ASTNodeVariant> get_predicate() final;

        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const std::vector<ASTNodeVariant> &dests() const { return mDdests; }

        [[nodiscard]] const ASTNodeVariant &expr() const { return mExpr; };
    };

} // namespace voila::ast