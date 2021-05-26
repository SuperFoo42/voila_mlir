#pragma once
#include "DotVisualizer.hpp"
#include "TypeInferer.hpp"
#include "ast/Fun.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace voila
{
    using namespace ast;
    class Program
    {

        std::unordered_map<std::string, Expression> func_vars;

      public:
        std::vector<std::unique_ptr<Fun>> functions;
        TypeInferer inferer;

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
        void set_main_args_shape(const std::unordered_map<std::string, size_t> &shapes);
        void set_main_args_type(const std::unordered_map<std::string, DataType> &types);
    };
} // namespace voila
