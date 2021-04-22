#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
    template<class NodeType>
    class BinaryOP : public IExpression
    {
      public:
        BinaryOP(NodeType &lhs, NodeType &rhs) : IExpression(), lhs{lhs}, rhs{rhs}
        {
			check(lhs, rhs);
            // TODO: check that lhs and rhs are numeric
        }

        virtual ~BinaryOP() = default;

        bool is_binary() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "binary operation";
        }

        void print(std::ostream &o) const override
        {
            o << lhs << type2string() << rhs;
        }

      protected:
        NodeType lhs, rhs;
	
		virtual void checkArgs(NodeType &lhs, NodeType &rhs) = 0;
    };
}