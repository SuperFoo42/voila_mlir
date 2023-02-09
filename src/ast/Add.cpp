#include "ast/Add.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast {
    bool Add::is_add() const {
        return true;
    }

    Add *Add::as_add() {
        return this;
    }

    std::string Add::type2string() const {
        return "add";
    }

    void Add::visit(ASTVisitor &visitor) const {
        visitor(*this);
    }

    void Add::visit(ASTVisitor &visitor) {
        visitor(*this);
    }
} // namespace voila::ast