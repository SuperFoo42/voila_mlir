#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
	template<class NodeType>
    class UnaryOP : virtual public IExpression
    {
      public:
        explicit UnaryOP(NodeType &param) : IExpression(), param{param}
        {
            checkArg(param);
        }

        ~UnaryOP() override = default;

        [[nodiscard]] bool is_unary() const final
        {
            return true;
        }

        [[nodiscard]] std::string type2string() const override
        {
            return "unary operation";
        }

        void print(std::ostream &o) const override
        {
            o << type2string() << param;
        }

      protected:
        NodeType param;
        virtual void checkArg(const NodeType &param) = 0;
    };
}