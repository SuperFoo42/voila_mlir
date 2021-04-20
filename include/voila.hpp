#pragma once
#include "ast/ASTNode.hpp"
#include "ast/Arithmetic.hpp"
#include "ast/BinaryOP.hpp"
#include "ast/Expression.hpp"
#include "ast/Statement.hpp"
#include "ast/UnaryOP.hpp"
#include "blend_space_point.hpp"

#include <cmath>
#include <concepts>
#include <magic_enum.hpp>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
// VOILA
namespace voila::ast
{
    // Expressions
    class Add : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "add";
        }

        bool is_add() const final
        {
            return true;
        }
    };

    class Sub : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "sub";
        }

        bool is_sub() const final
        {
            return true;
        }
    };

    class Mul : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "mul";
        }

        bool is_mul() const final
        {
            return true;
        }
    };

    class Div : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "div";
        }

        bool is_div() const final
        {
            return true;
        }
    };

    class Mod : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "mod";
        }

        bool is_mod() const final
        {
            return true;
        }
    };

    struct Comparison : ASTNode
    {
        virtual ~Comparison() = default;

        bool is_comparison() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "comparison";
        }
    };

    class Eq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "eq";
        }

        bool is_eq() const final
        {
            return true;
        }
    };

    class Neq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;
        std::string type2string() const final
        {
            return "neq";
        }

        bool is_neq() const final
        {
            return true;
        }
    };

    class Le : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "le";
        }

        bool is_le() const final
        {
            return true;
        }
    };

    class Leq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "leq";
        }

        bool is_leq() const final
        {
            return true;
        }
    };

    class Ge : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "ge";
        }

        bool is_ge() const final
        {
            return true;
        }
    };

    class Geq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "geq";
        }

        bool is_geq() const final
        {
            return true;
        }
    };

    struct Logical : ASTNode
    {
        virtual ~Logical() = default;

        bool is_logical() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "logical";
        }
    };

    class And : BinaryOP<Expression>, Logical
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "and";
        }

        bool is_and() const final
        {
            return true;
        }
    };

    class Or : BinaryOP<Expression>, Logical
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "or";
        }

        bool is_or() const final
        {
            return true;
        }
    };

    class Not : UnaryOP<Expression>, Logical
    {
      public:
        using UnaryOP::param;
        using UnaryOP::UnaryOP;

        std::string type2string() const final
        {
            return "not";
        }

        bool is_not() const final
        {
            return true;
        }
    };

    // Statements
    class Loop : ASTNode
    {
      public:
        Loop(const Expression &pred, const std::vector<Statement> &stms) : ASTNode(), pred{pred}, stms{stms} {}

        std::string type2string() const final
        {
            return "loop";
        }

        bool is_loop() const final
        {
            return true;
        }

        Expression pred;
        std::vector<Statement> stms;
        // TODO
        // CrossingVariables crossing_variables;
    };


    struct Assign : ASTNode
    {
        Assign(const std::string &dest, const ExprPtr &expr, const ExprPtr &pred) :
            Statement(Statement::Type::Assignment, dest, expr, pred)
        {
        }
    };

    struct DCol
    {
        enum Modifier
        {
            kValue,
            kKey,
            kHash
        };

        const std::string name;
        const std::string source; // only for columns of BaseTables
        const Modifier mod;

        DCol(const std::string &name, const std::string &source = "", Modifier mod = Modifier::kValue) :
            name(name), source(source), mod(mod)
        {
        }
    };

    struct DataStructure
    {
        enum Type
        {
            kTable,
            kHashTable,
            kBaseTable
        };

        typedef int Flags;
        static constexpr Flags kThreadLocal = 1 << 1;
        static constexpr Flags kReadAfterWrite = 1 << 2;
        static constexpr Flags kFlushToMaster = 1 << 3;
        static constexpr Flags kDefault = 0;

        static std::string type_to_str(Type t);

        const std::string name;
        const std::string source; // only for BaseTables
        const Type type;
        const Flags flags;

        const std::vector<DCol> cols;

        DataStructure(const std::string &name,
                      const Type &type,
                      const Flags &flags,
                      const std::vector<DCol> &cols,
                      const std::string &source = "") :
            name(name), source(source), type(type), flags(flags), cols(cols)
        {
        }
    };

    struct Program
    {
        std::vector<Pipeline> pipelines;
        std::vector<DataStructure> data_structures;

        // Program(std::vector<Pipeline>& pipelines) : pipelines(pipelines) {}
    };

    struct CrossingVariables
    {
        // all inputs going across this statement
        std::unordered_set<std::string> all_inputs;
        // all outputs going across this statement
        std::unordered_set<std::string> all_outputs;

        // used inside the statement
        std::unordered_set<std::string> used_inputs;

        // generated/update inside the statement
        std::unordered_set<std::string> used_outputs;
    };

    struct WrapStatements : Statement
    {
        WrapStatements(const std::vector<std::shared_ptr<Statement>> &stms, const ExprPtr &pred) :
            Statement(Statement::Type::Wrap, stms, "", pred)
        {
        }
    };


    struct Effect : Statement
    {
        Effect(const ExprPtr &expr) : Statement(Statement::Type::EffectExpr, "", expr) {}
    };

    struct Emit : Statement
    {
        Emit(const ExprPtr &expr,
             const ExprPtr &pred); // TODO
    };

    struct TupleCreate : Expression
    {
        TupleCreate(std::vector<ExprPtr> &tupleElems) : Expression(NoPred, "create tuple", tupleElems) {}
    };

    struct LolePred : Expression
    {
        LolePred() : Expression(Expression::Type::LolePred, "") {}
    };

    struct Const : Expression
    {
        Const(const std::string &val) : Expression(Expression::Type::Constant, val) {}
        Const(int val) : Expression(Expression::Type::Constant, std::to_string(val)) {}
        Const(bool val) : Expression(Expression::Type::Constant, std::to_string(val)) {}
    };

    struct LoleArg : Expression
    {
        LoleArg() : Expression(Expression::Type::LoleArg, "") {}
    };

    struct Fun : Expression
    {
        Fun(const std::string &fun, const std::vector<ExprPtr> &exprs, const ExprPtr &pred) :
            Expression(Expression::Type::Function, fun, exprs, pred)
        {
        }

        virtual ~Fun() {}
    };

    struct FunctionCall : Expression
    {
        FunctionCall(const std::string fun, std::vector<ExprPtr> args) :
            Expression(Expression::Type::FunctionCall, fun, args)
        {
        }
        virtual ~FunctionCall() = default;
    };

    struct Main : Fun
    {
        Main(const std::vector<ExprPtr> &exprs) : Fun("main", exprs, nullptr) {}

        virtual ~Main() {}
    };

    struct Read : Fun
    {
        Read(const ExprPtr &col, const ExprPtr &readPos) : Fun("read", {col, readPos}, nullptr) {}

        virtual ~Read() {}
    };

    struct Gather : Fun
    {
        Gather(const ExprPtr &col, const ExprPtr &readIdxs) : Fun("gather", {col, readIdxs}, nullptr) {}

        virtual ~Gather() {}
    };

    struct TupleAppend : Fun
    {
        TupleAppend(const std::vector<ExprPtr> &expr, const ExprPtr &pred) : Fun("tappend", expr, pred) {}
    };

    struct BlendStmt : Statement
    {
        BlendStmt(const StmtList &stms, const ExprPtr &pred) : Statement(Statement::Type::BlendStmt, stms, "", pred) {}

        BlendConfigPtr blend;

        CrossingVariables crossing_variables;

        ExprPtr get_predicate() const;
    };

    struct MetaStmt : Statement
    {
        enum MetaType
        {
            VarDead,
            RefillInflow,
            FsmExclusive,
        };

        const MetaType meta_type;

        MetaStmt(MetaType t) : Statement(Statement::Type::MetaStmt, {}, "", nullptr), meta_type(t) {}
    };

    struct MetaVarDead : MetaStmt
    {
        std::string variable_name;
        MetaVarDead(const std::string &var) : MetaStmt(MetaStmt::MetaType::VarDead), variable_name(var) {}
    };

    struct MetaFsmExclusive : MetaStmt
    {
        const bool begin;

        MetaFsmExclusive(bool begin) : MetaStmt(MetaStmt::MetaType::FsmExclusive), begin(begin) {}
    };

    struct MetaBeginFsmExclusive : MetaFsmExclusive
    {
        MetaBeginFsmExclusive() : MetaFsmExclusive(true) {}
    };

    struct MetaEndFsmExclusive : MetaFsmExclusive
    {
        MetaEndFsmExclusive() : MetaFsmExclusive(false) {}
    };

    struct MetaRefillInflow : MetaStmt
    {
        MetaRefillInflow() : MetaStmt(MetaStmt::MetaType::RefillInflow) {}
    };

    struct TupleGet : Fun
    {
        static long long get_idx(Expression &e);

        TupleGet(const ExprPtr &e, size_t idx);
    };

    struct Print : Fun
    {
        Print(const ExprPtr &col, const ExprPtr &pred) : Fun("print", {col}, pred) {}
    };

    struct Scatter : Effect
    {
        Scatter(const ExprPtr &col, const ExprPtr &idx, const ExprPtr &val, const ExprPtr &pred) :
            Effect(std::make_shared<Fun>("scatter", ExprList{col, idx, val}, pred))
        {
        }
    };

    struct Write : Effect
    {
        Write(const ExprPtr &col, const ExprPtr &wpos, const ExprPtr &val, const ExprPtr &pred) :
            Effect(std::make_shared<Fun>("write", ExprList{col, wpos, val}, pred))
        {
        }
    };

    // FIXME
    struct Aggr2 : Effect
    {
        Aggr2(const ExprPtr &col,
              const ExprPtr &idx,
              const ExprPtr &val,
              const std::string &name,
              const ExprPtr &pred) :
            Effect(std::make_shared<Fun>(name, ExprList{col, idx, val}, pred))
        {
        }
    };

#define _Aggr2(external, internal)                                                                                     \
    struct external : Aggr2                                                                                            \
    {                                                                                                                  \
        external(const ExprPtr &a, const ExprPtr &b, const ExprPtr &c, const ExprPtr &p) : Aggr2(a, b, c, internal, p) \
        {                                                                                                              \
        }                                                                                                              \
    };
    _Aggr2(AggrCount, "aggr_count") _Aggr2(AggrSum, "aggr_sum") _Aggr2(AggrMin, "aggr_min") _Aggr2(AggrMax, "aggr_max")

        struct AggrGSum : Effect
    {
        AggrGSum(const ExprPtr &col, const ExprPtr &val, const ExprPtr &pred) :
            Effect(std::make_shared<Fun>("aggr_gsum", ExprList{col, val}, pred))
        {
        }
    };

    struct AggrGCount : Effect
    {
        AggrGCount(const ExprPtr &col, const ExprPtr &pred) :
            Effect(std::make_shared<Fun>("aggr_gcount", ExprList{col}, pred))
        {
        }
    };

    struct Ref : Expression
    {
        Ref(const std::string &var) : Expression(Expression::Type::Reference, var) {}
    };

    struct Table : DataStructure
    {
        Table(const std::string &name,
              const std::vector<DCol> &cols,
              const DataStructure::Type &type = DataStructure::kTable,
              const DataStructure::Flags &flags = DataStructure::kDefault);
    };

    struct HashTable : Table
    {
        HashTable(const std::string &name, const DataStructure::Flags &flags, const std::vector<DCol> &cols) :
            Table(name, cols, DataStructure::kHashTable, flags)
        {
        }
    };

    struct BaseTable : DataStructure
    {
        BaseTable(const std::string &name, const std::vector<DCol> &cols, const std::string &source) :
            DataStructure(name, DataStructure::kBaseTable, DataStructure::kDefault, cols, source)
        {
        }
    };

    // TODO: Which of this types do we need?
    struct ScalarTypeProps
    {
        double dmin = std::nan("0");
        double dmax = std::nan("1");
        std::string type = "";

        bool operator==(const ScalarTypeProps &o) const
        {
            return dmin == o.dmin && dmax == o.dmax && type == o.type;
        }

        bool operator!=(const ScalarTypeProps &other) const
        {
            return !operator==(other);
        }

        void write(std::ostream &o, const std::string &prefix = "") const;

        void from_cover(ScalarTypeProps &a, ScalarTypeProps &b);
    };

    struct TypeProps
    {
        enum Category
        {
            Unknown = 0,
            Tuple,
            Predicate
        };
        Category category = Unknown;
        std::vector<ScalarTypeProps> arity;

        bool operator==(const TypeProps &other) const
        {
            return category == other.category && arity == other.arity;
        }

        bool operator!=(const TypeProps &other) const
        {
            return !operator==(other);
        }

        void write(std::ostream &o, const std::string &prefix = "") const;

        void from_cover(TypeProps &a, TypeProps &b);
    };

    struct Properties
    {
        TypeProps type;

        bool scalar = false;
        bool constant = false;
        bool refers_to_variable = false; // set in LimitVariableLifetimeCheck

        std::string first_touch_random_access; // !empty when doing prefetch, also tells column to prefetch

        std::vector<ExprPtr> forw_affected_scans; // for scan_pos, later scanned columns

        std::vector<bool> gen_recurse;

        LoopReferencesProp loop_refs;

        std::shared_ptr<Expression> vec_cardinality;
    };

    struct LoopReferencesProp
    {
        std::unordered_set<std::string> ref_strings;

        void insert(const std::string &s);
    };

    struct Lolepop
    {
        const std::string name;
        std::vector<std::shared_ptr<Statement>> statements;

        Lolepop(const std::string &name, const std::vector<std::shared_ptr<Statement>> &statements) :
            name(name), statements(statements)
        {
        }
    };

    struct Pipeline
    {
        std::vector<std::shared_ptr<Lolepop>> lolepops;

        bool tag_interesting = true; //!< To focus exploration

        // Pipeline(std::vector<Lolepop*>& lolepops) : lolepops(lolepops) {}
    };

} // namespace voila::ast