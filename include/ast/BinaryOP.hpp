#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
    template<class NodeType>
    class BinaryOP : virtual public IExpression
    {
      public:
        BinaryOP(NodeType &lhs, NodeType &rhs) : IExpression(), lhs{lhs}, rhs{rhs}
        {
        }

        ~BinaryOP() override = default;

        [[nodiscard]] bool is_binary() const final
        {
            return true;
        }

        [[nodiscard]] std::string type2string() const override
        {
            return "binary operation";
        }

        void print(std::ostream &o) const override
        {
            o << lhs << type2string() << rhs;
        }

      protected:
        NodeType lhs, rhs;
    };
}