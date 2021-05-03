#include "Program.hpp"

#include <fstream>
namespace voila
{
    void Program::infer_type(const Expression &node) {
        node.visit(inferer);
    }
    void Program::infer_type(const ASTNode &node) {
        node.visit(inferer);
    }
    void Program::infer_type(const Statement &node) {
        node.visit(inferer);
    }


    void Program::to_dot(const std::string &fname)
    {
        for (auto &func : functions)
        {
            DotVisualizer vis(*func, std::optional<std::reference_wrapper<TypeInferer>>(inferer));
            std::ofstream out(fname + "." + func->name+".dot", std::ios::out);
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
} // namespace voila