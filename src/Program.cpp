#include "Program.hpp"

#include "MainFunctionNotFoundException.hpp"

#include <fstream>
namespace voila
{
    void Program::infer_type(const Expression &node)
    {
        node.visit(inferer);
    }

    void Program::infer_type(const ASTNode &node)
    {
        node.visit(inferer);
    }

    void Program::infer_type(const Statement &node)
    {
        node.visit(inferer);
    }

    void Program::to_dot(const std::string &fname)
    {
        for (auto &func : functions)
        {
            DotVisualizer vis(*func, std::optional<std::reference_wrapper<TypeInferer>>(inferer));
            std::ofstream out(fname + "." + func->name + ".dot", std::ios::out);
            out << vis;
            out.close();
        }
    }

    bool Program::has_var(const std::string &var_name)
    {
        return func_vars.contains(var_name);
    }

    Expression Program::get_var(const std::string &var_name)
    {
        return func_vars.at(var_name);
    }

    void Program::add_var(Expression expr)
    {
        assert(expr.is_variable());
        func_vars.emplace(expr.as_variable()->var, expr);
    }

    void Program::set_main_args_shape(const std::unordered_map<std::string, size_t> &shapes)
    {
        const auto main = std::find_if(
            functions.begin(), functions.end(), [](const auto &f) -> auto { return dynamic_cast<Main *>(f.get()); });
        if (main == functions.end())
            throw MainFunctionNotFoundException();
        auto &args = (*main)->args;
        for (auto &arg : args)
        {
            assert(arg.is_variable());
            if (shapes.contains(arg.as_variable()->var))
            {
                inferer.set_arity(arg.as_expr(), shapes.at(arg.as_variable()->var));
            }
        }
    }

    void Program::set_main_args_type(const std::unordered_map<std::string, DataType> &types)
    {
        const auto main = std::find_if(
            functions.begin(), functions.end(), [](const auto &f) -> auto { return dynamic_cast<Main *>(f.get()); });
        if (main == functions.end())
            throw MainFunctionNotFoundException();
        auto &args = (*main)->args;
        for (auto &arg : args)
        {
            assert(arg.is_variable());
            if (types.contains(arg.as_variable()->var))
            {
                inferer.set_type(arg.as_expr(), types.at(arg.as_variable()->var));
            }
        }
    }
} // namespace voila