#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr, enable_shared_from_this
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Variable : public AbstractASTNode<Variable>, virtual public std::enable_shared_from_this<Variable>
    {
      public:
        // TODO: private ctor to mitigate stack allocations and resulting getptr nullptr
        explicit Variable(const Location loc, std::string val) : AbstractASTNode<Variable>(loc), var{std::move(val)} {}

        [[nodiscard]] std::string type2string_impl() const { return "variable"; };
        void print_impl(std::ostream &ostream) const { ostream << var; };

        std::shared_ptr<Variable> getptr() { return shared_from_this(); }

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        const std::string var;
    };
} // namespace voila::ast