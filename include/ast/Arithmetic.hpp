#pragma once
#include "ASTNodeVariant.hpp"
#include "IExpression.hpp"     // for IExpression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for make_shared, shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class Arithmetic : public IExpression
    {
        ASTNodeVariant mLhs, mRhs;

      public:
        Arithmetic(Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs);
        ~Arithmetic() override = default;

        [[nodiscard]] bool is_arithmetic() const final;

        Arithmetic *as_arithmetic() final;

        [[nodiscard]] std::string type2string() const override;
        void print(std::ostream &ostream) const final;

        [[nodiscard]] const ASTNodeVariant &lhs() const { return mLhs; }

        [[nodiscard]] const ASTNodeVariant &rhs() const { return mRhs; }

      protected:
        template <class T>
        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
            requires std::is_base_of_v<Arithmetic, T>
        {
            auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                           [](std::monostate &) -> ASTNodeVariant  { throw std::logic_error(""); }};
            return std::make_shared<T>(loc, std::visit(cloneVisitor, mLhs), std::visit(cloneVisitor, mRhs));
        }
    };
} // namespace voila::ast