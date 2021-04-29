#pragma once
#include "location.hpp"

#include <bit>
#include <climits>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
namespace voila::ast
{
    class ASTVisitor;
    class Fun;
    class Main;
    using Location = voila::parser::location;
    class ASTNode
    {
      public:
        Location loc;

        virtual void print(std::ostream &) const = 0;
        [[nodiscard]] virtual std::string type2string() const = 0;
        virtual void visit(ASTVisitor &visitor) const;
        virtual void visit(ASTVisitor &visitor);
        virtual ~ASTNode() = default;
        explicit ASTNode(Location loc);
        ASTNode();

        // ASTNode() = default;

        [[nodiscard]] Location get_location() const;

        [[nodiscard]] virtual bool is_expr() const;

        [[nodiscard]] virtual bool is_stmt() const;

        [[nodiscard]] virtual bool is_function_definition() const;

        [[nodiscard]] virtual bool is_main() const;

        virtual Fun *as_function_definition();

        virtual Main *as_main();

        virtual bool operator==(const ASTNode &rhs) const;
        virtual bool operator!=(const ASTNode &rhs) const;
    };
} // namespace voila::ast

namespace std
{
    inline void hash_combine(std::size_t &) {}

    template<class T, class... Rest>
    inline void hash_combine(std::size_t &seed, const T &v, Rest... rest)
    {
        std::hash<T> hasher;
        seed ^= std::rotl(hasher(v), sizeof(size_t) * CHAR_BIT / 2);
        hash_combine(seed, rest...);
    }

    template<>
    struct hash<voila::ast::ASTNode>
    {
        std::size_t operator()(voila::ast::ASTNode const &node)
        {
            std::size_t res = 0;
            hash_combine(res, *node.loc.begin.filename, *node.loc.end.filename, node.loc.begin.line, node.loc.end.line,
                         node.loc.begin.column, node.loc.end.column);
            return res;
        }
    };
} // namespace std