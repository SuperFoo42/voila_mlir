#pragma once
#include "ASTNodeVariant.hpp"
#include "location.hpp" // for position, location
#include <bit>          // for rotl
#include <climits>      // for CHAR_BIT
#include <cstddef>      // for size_t
#include <functional>   // for hash
#include <iostream>     // for ostream
#include <memory>       // for shared_ptr
#include <string>       // for string
#include <variant>      // for hash

namespace voila::ast
{
    template <class T> class ASTVisitor;

    using Location = voila::parser::location;

    template <class ASTNode> class AbstractASTNode
    {
      public:
        Location loc;

        void print(std::ostream &o) const { static_cast<const ASTNode *>(this)->print_impl(o); }

        [[nodiscard]] std::string type2string() const { return static_cast<const ASTNode *>(this)->type2string_impl(); }

        ~AbstractASTNode() = default;

        explicit AbstractASTNode(Location loc) : loc(loc) {}

        [[nodiscard]] Location get_location() const { return loc; }

        bool operator==(const ASTNode &rhs) const
        {
            return *loc.begin.filename == *rhs.loc.begin.filename && loc.begin.column == rhs.loc.begin.column &&
                   loc.begin.line == rhs.loc.begin.line && loc.end.filename == rhs.loc.end.filename &&
                   loc.end.column == rhs.loc.end.column && loc.end.line == rhs.loc.end.line;
        }

        bool operator!=(const ASTNode &rhs) const { return !(*static_cast<ASTNode *>(this) == rhs); }

        ASTNodeVariant clone(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
        {
            return static_cast<ASTNode *>(this)->clone_impl(vmap);
        }
    };

} // namespace voila::ast

inline void hash_combine(std::size_t &) {}

template <class T, class... Rest> inline void hash_combine(std::size_t &seed, const T &v, Rest... rest)
{
    std::hash<T> hasher;
    seed ^= std::rotl(hasher(v), sizeof(size_t) * CHAR_BIT / 2);
    hash_combine(seed, rest...);
}

template <class Derived> struct std::hash<voila::ast::AbstractASTNode<Derived>>
{
    std::size_t operator()(voila::ast::AbstractASTNode<Derived> const &node)
    {
        std::size_t res = 0;
        hash_combine(res, *node.loc.begin.filename, *node.loc.end.filename, node.loc.begin.line, node.loc.end.line,
                     node.loc.begin.column, node.loc.end.column);
        return res;
    }
};

template <class Derived> std::ostream &operator<<(std::ostream &out, const voila::ast::AbstractASTNode<Derived> &t)
{
    t.print(out);
    return out;
}