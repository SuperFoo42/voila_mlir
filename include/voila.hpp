#pragma once
#include "ast/ASTNode.hpp"
#include "ast/Arithmetic.hpp"
#include "ast/BinaryOP.hpp"
#include "ast/Expression.hpp"
#include "ast/IExpression.hpp"
#include "ast/Statement.hpp"
#include "ast/UnaryOP.hpp"
#include "blend_space_point.hpp"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <magic_enum.hpp>
#include <memory>
#include <optional>
#include <string>
#include <tbb/internal/_aggregator_impl.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <valarray>
#include <vector>
// VOILA
namespace voila::ast
{
    // Statements

    // TODO: implicit?

    // TODO: Which of this types do we need?
    /*

    // FIXME
    struct Aggr2 : Effect
    {
        Aggr2(const ExprPtr &col, const ExprPtr &idx, const ExprPtr &val, const std::string &name, const ExprPtr &pred)
    : Effect(std::make_shared<Fun>(name, ExprList{col, idx, val}, pred))
        {
        }
    };

    #define _Aggr2(external, internal) \
        struct external : Aggr2 \
        { \
            external(const ExprPtr &a, const ExprPtr &b, const ExprPtr &c, const ExprPtr &p) : Aggr2(a, b, c, internal,
    p) \
            { \
            } \
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

    struct LolePred : Expression
    {
        LolePred() : Expression(Expression::Type::LolePred, "") {}
    };

    struct LoleArg : Expression
    {
        LoleArg() : Expression(Expression::Type::LoleArg, "") {}
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
    */
} // namespace voila::ast