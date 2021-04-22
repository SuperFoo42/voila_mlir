#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
	template<class NodeType>
    class UnaryOP : public IExpression
    {
      public:
        UnaryOP(NodeType &param) : IExpression(), param{param}
        {
            checkArg(param);
        }

        virtual ~UnaryOP() = default;

        bool is_unary() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "binary operation";
        }

        void print(std::ostream &o) const override
        {
            o << type2string() << param;
        }

      protected:
        NodeType param;
		virtual void checkArg(NodeType &param) = 0;
    };
}