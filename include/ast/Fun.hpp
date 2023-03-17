#pragma once

#include "ASTNode.hpp"
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <optional>            // for optional
#include <string>              // for string, hash
#include <unordered_map>
#include <vector>

namespace voila::ast
{
    class Fun : public AbstractASTNode
    {
        std::string mName;
        std::vector<ASTNodeVariant> mArgs;
        std::vector<ASTNodeVariant> mBody;
        ASTNodeVariant mResult;
        std::unordered_map<std::string, ASTNodeVariant> mVariables;

      public:
        Fun(Location loc, std::string fun, std::vector<ASTNodeVariant> args, std::vector<ASTNodeVariant> exprs);

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

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        const std::string &name() const { return mName; }

        const std::vector<ASTNodeVariant> &args() const { return mArgs; }

        const std::vector<ASTNodeVariant> &body() const { return mBody; }

        const ASTNodeVariant &result() const
        {
            return mResult;
        }

        std::unordered_map<std::string, ASTNodeVariant> &variables() { return mVariables; }

        const std::unordered_map<std::string, ASTNodeVariant> &variables() const { return mVariables; }
    };
} // namespace voila::ast