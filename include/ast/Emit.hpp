#pragma once
#include "range/v3/range/concepts.hpp"
#include "range/v3/range/conversion.hpp"
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{
    class Emit : public AbstractASTNode<Emit>
    {
        std::vector<ASTNodeVariant> mExprs;

      public:
        explicit Emit(Location loc, ranges::input_range auto && expr)
            : AbstractASTNode(loc), mExprs{ranges::to<std::vector>(expr)}
        {
        }

        [[nodiscard]] std::string type2string_impl() const;

        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const std::vector<ASTNodeVariant> &exprs() const { return mExprs; }
    };

} // namespace voila::ast