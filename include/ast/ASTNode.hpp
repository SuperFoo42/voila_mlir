#pragma once
#include "location.hpp"        // for position, location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <bit>                 // for rotl
#include <climits>             // for CHAR_BIT
#include <cstddef>             // for size_t
#include <functional>          // for hash
#include <iostream>            // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <variant>             // for hash
#include "ASTNodeVariant.hpp"

namespace voila::ast
{
    template <class T> class ASTVisitor;

    using Location = voila::parser::location;

    class AbstractASTNode
    {
      public:
        Location loc;

        virtual void print(std::ostream &) const = 0;

        [[nodiscard]] virtual std::string type2string() const = 0;

        virtual ~AbstractASTNode() = default;

        explicit AbstractASTNode(Location loc);

        AbstractASTNode();

        [[nodiscard]] Location get_location() const;

        [[nodiscard]] virtual bool is_expr() const;

        [[nodiscard]] virtual bool is_stmt() const;

        [[nodiscard]] virtual bool is_function_definition() const;

        [[nodiscard]] virtual bool is_main() const;

        virtual Fun *as_function_definition();

        virtual Main *as_main();

        virtual bool operator==(const AbstractASTNode &rhs) const;

        virtual bool operator!=(const AbstractASTNode &rhs) const;

        virtual ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) = 0;
    };

} // namespace voila::ast

inline void hash_combine(std::size_t &) {}

template <class T, class... Rest> inline void hash_combine(std::size_t &seed, const T &v, Rest... rest)
{
    std::hash<T> hasher;
    seed ^= std::rotl(hasher(v), sizeof(size_t) * CHAR_BIT / 2);
    hash_combine(seed, rest...);
}

template <> struct std::hash<voila::ast::AbstractASTNode>
{
    std::size_t operator()(voila::ast::AbstractASTNode const &node);
};

std::ostream &operator<<(std::ostream &out, const voila::ast::AbstractASTNode &t);