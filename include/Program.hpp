#pragma once
#include "DotVisualizer.hpp"
#include "TypeInferer.hpp"
#include "ast/Fun.hpp"

namespace voila
{
    using namespace ast;
    class Program
    {
        std::vector<std::unique_ptr<Fun>> functions;
        TypeInferer inferer;
        std::unordered_map<std::string, Expression> func_vars;

      public:
        Program() = default;

        void add_func(Fun *f)
        {
            functions.emplace_back(f);
            f->variables = std::move(func_vars);
            func_vars.clear();
        }

        std::vector<std::unique_ptr<Fun>> &get_funcs()
        {
            return functions;
        }

        void infer_type(const ASTNode &node);
        void infer_type(const Expression &node);
        void infer_type(const Statement &node);

        void to_dot(const std::string &);

        void add_var(Expression expr);

        Expression get_var(const std::string& var_name);

        bool has_var(const std::string& var_name);
    };
} // namespace voila
