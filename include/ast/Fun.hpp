#pragma once

#include "ASTNode.hpp"
#include "Statement.hpp"
#include "ast/Expression.hpp"  // for Expression
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <optional>            // for optional
#include <string>              // for string, hash
#include <unordered_map>
#include <vector>

namespace voila::ast
{
    class ASTVisitor;

    class Fun : public ASTNode
    {
        std::string mName;
        std::vector<Expression> mArgs;
        std::vector<Statement> mBody;
        std::optional<Statement> mResult;
        std::unordered_map<std::string, Expression> mVariables;

      public:
        Fun(Location loc, std::string fun, std::vector<Expression> args, std::vector<Statement> exprs);

        Fun() = default;

        Fun(Fun &) = default;

        Fun(const Fun &) = default;

        Fun(Fun &&) = default;

        Fun &operator=(const Fun &) = default;

        ~Fun() override = default;

        [[nodiscard]] bool is_function_definition() const override;

        Fun *as_function_definition() override;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &o) const override;

        void visit(ASTVisitor &visitor) const override;

        void visit(ASTVisitor &visitor) override;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const std::string &name() const { return mName; }

        const std::vector<Expression> &args() const { return mArgs; }

        const std::vector<Statement> &body() const { return mBody; }

        const std::optional<Statement> &result() const { return mResult; }

        std::unordered_map<std::string, Expression> &variables() { return mVariables; }

        const std::unordered_map<std::string, Expression> &variables() const { return mVariables; }
    };
} // namespace voila::ast