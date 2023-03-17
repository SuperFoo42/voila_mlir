#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <utility>              // for move
#include <vector>               // for vector
#include "IExpression.hpp"      // for IExpression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class Insert : public IExpression
    {
        ASTNodeVariant mKeys;
        std::vector<ASTNodeVariant> mValues;

      public:
        Insert(Location loc, ASTNodeVariant keys, std::vector<ASTNodeVariant> values) :
            IExpression(loc), mKeys{std::move(keys)}, mValues{std::move(values)}
        {
            // TODO
        }

        [[nodiscard]] bool is_insert() const final;

        Insert *as_insert() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &keys() const {
            return mKeys;
        }

        [[nodiscard]] const std::vector<ASTNodeVariant> &values() const {
            return mValues;
        }
    };
} // namespace voila::ast