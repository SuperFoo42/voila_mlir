#pragma once
#include "ASTNode.hpp"

namespace voila::ast
{
	template<class NodeType>
    class UnaryOP : ASTNode
    {
      public:
        UnaryOP(NodeType &param) : ASTNode(), param{param}
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