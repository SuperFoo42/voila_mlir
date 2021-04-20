#include "ast/ASTNode.hpp"
#include "ast/visitor.hpp"

namespace voila::ast {
void ASTNode::visit(ASTVisitor &visitor) {
    visitor(*this);
}
}