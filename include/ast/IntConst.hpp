#pragma once

#include <cstdint>              // for intmax_t
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include "Const.hpp"            // for Const
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast {

    class ASTVisitor;

    class IntConst : public Const {

    public:
        explicit IntConst(Location loc, const std::intmax_t val) : Const(loc), val{val} {}

        [[nodiscard]] bool is_integer() const final;

        IntConst *as_integer() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &) override;

        const std::intmax_t val;
    };
} // namespace voila::ast