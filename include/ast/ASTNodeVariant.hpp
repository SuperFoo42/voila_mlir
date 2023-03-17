#pragma once
#include <memory>
#include <variant>

namespace voila::ast
{
    class Fun;
    class Add;
    class AggrAvg;
    class AggrCnt;
    class AggrMax;
    class AggrMin;
    class AggrSum;
    class And;
    class Assign;
    class BooleanConst;
    class Div;
    class Emit;
    class Eq;
    class FltConst;
    class FunctionCall;
    class Gather;
    class Ge;
    class Geq;
    class Hash;
    class Insert;
    class IntConst;
    class Le;
    class Leq;
    class Loop;
    class Mod;
    class Mul;
    class Neq;
    class Not;
    class Or;
    class Predicate;
    class Read;
    class Ref;
    class Scatter;
    class Selection;
    class StatementWrapper;
    class StrConst;
    class Sub;
    class Variable;
    class Write;
    class Lookup;

    class Main;

    struct ASTNodeVariant : public std::variant<std::monostate,
                                                std::shared_ptr<Fun>,
                                                std::shared_ptr<Main>,
                                                std::shared_ptr<Add>,
                                                std::shared_ptr<AggrAvg>,
                                                std::shared_ptr<AggrCnt>,
                                                std::shared_ptr<AggrMax>,
                                                std::shared_ptr<AggrMin>,
                                                std::shared_ptr<AggrSum>,
                                                std::shared_ptr<And>,
                                                std::shared_ptr<Assign>,
                                                std::shared_ptr<BooleanConst>,
                                                std::shared_ptr<Div>,
                                                std::shared_ptr<Emit>,
                                                std::shared_ptr<Eq>,
                                                std::shared_ptr<FltConst>,
                                                std::shared_ptr<FunctionCall>,
                                                std::shared_ptr<Gather>,
                                                std::shared_ptr<Ge>,
                                                std::shared_ptr<Geq>,
                                                std::shared_ptr<Hash>,
                                                std::shared_ptr<Insert>,
                                                std::shared_ptr<IntConst>,
                                                std::shared_ptr<Le>,
                                                std::shared_ptr<Leq>,
                                                std::shared_ptr<Loop>,
                                                std::shared_ptr<Lookup>,
                                                std::shared_ptr<Mod>,
                                                std::shared_ptr<Mul>,
                                                std::shared_ptr<Neq>,
                                                std::shared_ptr<Not>,
                                                std::shared_ptr<Or>,
                                                std::shared_ptr<Predicate>,
                                                std::shared_ptr<Read>,
                                                std::shared_ptr<Ref>,
                                                std::shared_ptr<Scatter>,
                                                std::shared_ptr<Selection>,
                                                std::shared_ptr<StatementWrapper>,
                                                std::shared_ptr<StrConst>,
                                                std::shared_ptr<Sub>,
                                                std::shared_ptr<Variable>,
                                                std::shared_ptr<Write>>
    {
        using variant::variant;
        explicit operator bool() const { return std::holds_alternative<std::monostate>(*this); }
    };

    template <class... Ts> struct overloaded : Ts...
    {
        using Ts::operator()...;
    };
    template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

} // namespace voila::ast

template <> struct std::hash<voila::ast::ASTNodeVariant>
{
    std::size_t operator()(voila::ast::ASTNodeVariant const &node) const
    {
        std::hash<std::variant<
            std::monostate, std::shared_ptr<voila::ast::Fun>, std::shared_ptr<voila::ast::Main>,
            std::shared_ptr<voila::ast::Add>, std::shared_ptr<voila::ast::AggrAvg>,
            std::shared_ptr<voila::ast::AggrCnt>, std::shared_ptr<voila::ast::AggrMax>,
            std::shared_ptr<voila::ast::AggrMin>, std::shared_ptr<voila::ast::AggrSum>,
            std::shared_ptr<voila::ast::And>, std::shared_ptr<voila::ast::Assign>,
            std::shared_ptr<voila::ast::BooleanConst>, std::shared_ptr<voila::ast::Div>,
            std::shared_ptr<voila::ast::Emit>, std::shared_ptr<voila::ast::Eq>, std::shared_ptr<voila::ast::FltConst>,
            std::shared_ptr<voila::ast::FunctionCall>, std::shared_ptr<voila::ast::Gather>,
            std::shared_ptr<voila::ast::Ge>, std::shared_ptr<voila::ast::Geq>, std::shared_ptr<voila::ast::Hash>,
            std::shared_ptr<voila::ast::Insert>, std::shared_ptr<voila::ast::IntConst>, std::shared_ptr<voila::ast::Le>,
            std::shared_ptr<voila::ast::Leq>, std::shared_ptr<voila::ast::Loop>, std::shared_ptr<voila::ast::Lookup>,
            std::shared_ptr<voila::ast::Mod>, std::shared_ptr<voila::ast::Mul>, std::shared_ptr<voila::ast::Neq>,
            std::shared_ptr<voila::ast::Not>, std::shared_ptr<voila::ast::Or>, std::shared_ptr<voila::ast::Predicate>,
            std::shared_ptr<voila::ast::Read>, std::shared_ptr<voila::ast::Ref>, std::shared_ptr<voila::ast::Scatter>,
            std::shared_ptr<voila::ast::Selection>, std::shared_ptr<voila::ast::StatementWrapper>,
            std::shared_ptr<voila::ast::StrConst>, std::shared_ptr<voila::ast::Sub>,
            std::shared_ptr<voila::ast::Variable>, std::shared_ptr<voila::ast::Write>>>
            hasher;
        return hasher(node);
    }
};
