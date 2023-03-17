#include "ast/Aggregation.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ASTNodes.hpp"
namespace voila::ast
{
    bool Aggregation::is_aggr() const { return true; }

    Aggregation *Aggregation::as_aggr() { return this; }

    std::string Aggregation::type2string() const { return "aggregation"; }
} // namespace voila::ast