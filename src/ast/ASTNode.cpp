#include "ast/ASTNode.hpp"

#include "ast/ASTVisitor.hpp"

namespace voila::ast
{
    bool AbstractASTNode::is_stmt() const
    {
        return false;
    }
    bool AbstractASTNode::is_function_definition() const
    {
        return false;
    }
    bool AbstractASTNode::is_main() const
    {
        return false;
    }
    Fun *AbstractASTNode::as_function_definition()
    {
        return nullptr;
    }
    Main *AbstractASTNode::as_main()
    {
        return nullptr;
    }
    bool AbstractASTNode::is_expr() const
    {
        return false;
    }

    Location AbstractASTNode::get_location() const
    {
        return loc;
    }
    AbstractASTNode::AbstractASTNode(const Location loc) : loc(loc) {}
    bool AbstractASTNode::operator==(const AbstractASTNode &rhs) const
    {
        return *loc.begin.filename == *rhs.loc.begin.filename && loc.begin.column == rhs.loc.begin.column &&
               loc.begin.line == rhs.loc.begin.line && loc.end.filename == rhs.loc.end.filename &&
               loc.end.column == rhs.loc.end.column && loc.end.line == rhs.loc.end.line;
    }
    bool AbstractASTNode::operator!=(const AbstractASTNode &rhs) const
    {
        return !(rhs == *this);
    }

    AbstractASTNode::AbstractASTNode() = default;
} // namespace voila::ast
std::ostream &operator<<(std::ostream &out, const voila::ast::AbstractASTNode &t)
{
    t.print(out);
    return out;
}


std::size_t std::hash<voila::ast::AbstractASTNode>::operator()(const voila::ast::AbstractASTNode &node)
{
    std::size_t res = 0;
    hash_combine(res, *node.loc.begin.filename, *node.loc.end.filename, node.loc.begin.line, node.loc.end.line,
                 node.loc.begin.column, node.loc.end.column);
    return res;
}