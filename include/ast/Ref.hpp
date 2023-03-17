#pragma once

#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string

#include "IExpression.hpp"      // for IExpression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap
#include "ASTNodeVariant.hpp"

namespace voila::ast {

    class Ref : public IExpression {
        ASTNodeVariant mRef;

    public:
        explicit Ref(Location loc, ASTNodeVariant ref);

        [[nodiscard]] bool is_reference() const final;

        [[nodiscard]] std::string type2string() const override;

        [[nodiscard]] const Ref *as_reference() const final;

        void print(std::ostream &o) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &ref() const {
            return mRef;
        }
    };
} // namespace voila::ast