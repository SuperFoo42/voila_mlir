#include "ast/Aggregation.hpp"

namespace voila::ast {
    bool Aggregation::is_aggr() const {
        return true;
    }

    Aggregation *Aggregation::as_aggr() {
        return this;
    }

    std::string Aggregation::type2string() const {
        return "aggregation";
    }

    std::unique_ptr<ASTNode> Aggregation::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        if (groups.has_value())
            return std::make_unique<Aggregation>(loc, src.clone(vmap), groups->clone(vmap));
        else
            return std::make_unique<Aggregation>(loc, src.clone(vmap));
    }
} // namespace voila::ast