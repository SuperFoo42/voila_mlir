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
    class Load;

    class Main;

    using ast_variant_t = std::variant<std::monostate,
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
                                       std::shared_ptr<Write>,
                                       std::shared_ptr<Load>>;

    struct ASTNodeVariant : public ast_variant_t
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
        std::hash<voila::ast::ast_variant_t> hasher;
        return hasher(node);
    }
};
