#include "ast/Leq.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast {
    std::string Leq::type2string() const {
        return "leq";
    }

    bool Leq::is_leq() const {
        return true;
    }

    Leq *Leq::as_leq() {
        return this;
    }

    void Leq::visit(ASTVisitor &visitor) const {
        visitor(*this);
    }

    void Leq::visit(ASTVisitor &visitor) {
        visitor(*this);
    }
} // namespace voila::ast