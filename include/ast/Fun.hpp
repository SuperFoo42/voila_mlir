#pragma once

#include "ASTNode.hpp"
#include "range/v3/all.hpp"
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <optional>            // for optional
#include <string>              // for string, hash
#include <unordered_map>
#include <vector>
#include "EmitNotLastStatementException.hpp"

namespace voila::ast
{
    class Fun : public AbstractASTNode<Fun>
    {
        std::string mName;
        std::vector<ASTNodeVariant> mArgs;
        std::vector<ASTNodeVariant> mBody;
        ASTNodeVariant mResult;
        std::unordered_map<std::string, ASTNodeVariant> mVariables;

      public:
        Fun(Location loc, std::string fun, ranges::input_range auto && args, ranges::input_range auto && exprs)
            : AbstractASTNode(loc),
              mName{std::move(fun)},
              mArgs{ranges::to<std::vector>(args)},
              mBody{ranges::to<std::vector>(exprs)},
              mResult{std::monostate()}
        {
            auto ret = std::find_if(mBody.begin(), mBody.end(),
                                    [](auto &e) -> auto { return std::holds_alternative<std::shared_ptr<Emit>>(e); });
            if (ret != mBody.end())
            {
                if (ret != mBody.end() - 1)
                {
                    throw EmitNotLastStatementException();
                }
                mResult = *ret;
            }
        }

        Fun(Fun &) = default;

        Fun(const Fun &) = default;

        Fun(Fun &&) = default;

        Fun &operator=(const Fun &) = default;

        virtual ~Fun() = default;

         [[nodiscard]] virtual std::string type2string_impl() const;

        void print_impl(std::ostream &o) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

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