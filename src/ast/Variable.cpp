#include "ast/Variable.hpp"

namespace voila::ast
{
    ASTNodeVariant Variable::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        if (vmap.contains(ASTNodeVariant(this->getptr())))
        {
            return vmap.at(ASTNodeVariant(this->getptr()));
        }

        auto res = std::make_shared<Variable>(loc, var);
        vmap.emplace(this->getptr(), res);
        return res;
    }
} // namespace voila::ast
