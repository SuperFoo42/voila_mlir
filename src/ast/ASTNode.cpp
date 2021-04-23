#include "ast/ASTNode.hpp"

#include "ast/ASTVisitor.hpp"

namespace voila::ast
{
    void ASTNode::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
    bool ASTNode::is_stmt() const
    {
        return false;
    }
    bool ASTNode::is_function_definition() const
    {
        return false;
    }
    bool ASTNode::is_main() const
    {
        return false;
    }
    Fun *ASTNode::as_function_definition()
    {
        return nullptr;
    }
    Main *ASTNode::as_main()
    {
        return nullptr;
    }
    bool ASTNode::is_expr() const
    {
        return false;
    }
} // namespace voila::ast