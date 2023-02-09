#pragma once
#include "Types.hpp"          // for DataType (ptr only), type_id_t, Type (...
#include <ast/ASTVisitor.hpp> // for ASTVisitor
#include <cstddef>            // for size_t
#include <memory>             // for shared_ptr
#include <unordered_map>      // for unordered_map
#include <utility>            // for pair
#include <vector>             // for vector

namespace voila
{
    class Program;

    namespace ast
    {
        class ASTNode;
        class Add;
        class AggrAvg;
        class AggrCnt;
        class AggrMax;
        class AggrMin;
        class AggrSum;
        class And;
        class Arithmetic;
        class Assign;
        class BooleanConst;
        class Comparison;
        class Div;
        class Emit;
        class Eq;
        class Expression;
        class FltConst;
        class Fun;
        class FunctionCall;
        class Gather;
        class Ge;
        class Geq;
        class Hash;
        class Insert;
        class IntConst;
        class Le;
        class Leq;
        class Lookup;
        class Loop;
        class Main;
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
        class Statement;
        class StatementWrapper;
        class StrConst;
        class Sub;
        class TupleCreate;
        class TupleGet;
        class Variable;
        class Write;
    } // namespace ast

    // TODO pimpl
    class TypeInferer : public ast::ASTVisitor
    {
      public:
        explicit TypeInferer(Program *prog) : prog(prog) {}

        void operator()(const ast::Write &write) final;
        void operator()(const ast::Scatter &scatter) final;
        void operator()(ast::FunctionCall &call) final;
        void operator()(const ast::Assign &assign) final;
        void operator()(const ast::Emit &emit) final;
        void operator()(const ast::Loop &loop) final;
        void operator()(const ast::Arithmetic &arithmetic) final;
        void operator()(const ast::Comparison &comparison) final;
        void operator()(const ast::IntConst &aConst) final;
        void operator()(const ast::BooleanConst &aConst) final;
        void operator()(const ast::FltConst &aConst) final;
        void operator()(const ast::StrConst &aConst) final;
        void operator()(const ast::Read &read) final;
        void operator()(const ast::Gather &gather) final;
        void operator()(const ast::Ref &param) final;
        void operator()(const ast::TupleGet &get) final;
        void operator()(const ast::TupleCreate &create) final;
        void operator()(const ast::Fun &fun) final;
        void operator()(const ast::Main &main) final;
        void operator()(const ast::Selection &selection) final;
        void operator()(const ast::Variable &var) final;

        void operator()(const ast::Add &var) final;
        void operator()(const ast::Sub &sub) final;
        void operator()(const ast::Mul &mul) final;
        void operator()(const ast::Div &div1) final;
        void operator()(const ast::Mod &mod) final;
        void operator()(const ast::AggrSum &sum) final;
        void operator()(const ast::AggrCnt &cnt) final;
        void operator()(const ast::AggrMin &aggrMin) final;
        void operator()(const ast::AggrMax &aggrMax) final;
        void operator()(const ast::AggrAvg &avg) final;
        void operator()(const ast::Eq &eq) final;
        void operator()(const ast::Neq &neq) final;
        void operator()(const ast::Le &le) final;
        void operator()(const ast::Ge &ge) final;
        void operator()(const ast::Leq &leq) final;
        void operator()(const ast::Geq &geq) final;
        void operator()(const ast::And &anAnd) final;
        void operator()(const ast::Or &anOr) final;
        void operator()(const ast::Not &aNot) final;
        void operator()(const ast::Predicate &pred) final;
        void operator()(const ast::StatementWrapper &wrapper) final;
        void operator()(const ast::Hash &hash) final;

        std::shared_ptr<Type> get_type(const ast::ASTNode &node) const;

        std::shared_ptr<Type> get_type(const ast::Expression &node) const;

        std::shared_ptr<Type> get_type(const ast::Statement &node) const;

        void set_arity(const ast::ASTNode *node, size_t ar);
        void set_type(const ast::ASTNode *node, DataType type);

        void insertNewType(const ast::ASTNode &node, DataType t, Arity ar);
        void insertNewTypeAs(const ast::ASTNode &node, const Type &t);
        void operator()(const ast::Lookup &lookup) override;
        void operator()(const ast::Insert &insert) override;

        std::vector<std::shared_ptr<Type>> types;
        Program *prog;

      private:
        void unify(ast::ASTNode &t1, ast::ASTNode &t2);
        void unify(const ast::Expression &t1, const ast::Expression &t2);
        void unify(const ast::Statement &t1, const ast::Statement &t2);

        static DataType convert(DataType, DataType);

        static bool compatible(DataType, DataType);

        /**
         * Check whether first type is convertible to second type
         * @param t1 Type to convert
         * @param t2 destination type
         * @return
         */
        static bool convertible(DataType t1, DataType t2);

        void
        insertNewFuncType(const ast::ASTNode &node, std::vector<size_t> typeParamIDs, DataType returnT, Arity returnAr);

        type_id_t get_type_id(const ast::Expression &node);
        type_id_t get_type_id(const ast::Statement &node);
        type_id_t get_type_id(const ast::ASTNode &node);
        void unify(const ast::ASTNode &t1, const ast::Statement &t2);
        void unify(ast::ASTNode &t1, ast::Expression &t2);
        void insertNewFuncType(const ast::ASTNode &node, std::vector<type_id_t> typeParamIDs, type_id_t returnTypeID);
        void insertNewFuncType(const ast::ASTNode &node,
                               std::vector<type_id_t> typeParamIDs,
                               const std::vector<std::pair<DataType, Arity>> &returnTypes);
        void insertNewFuncType(const ast::ASTNode &node,
                               std::vector<type_id_t> typeParamIDs,
                               std::vector<type_id_t> returnTypeIDs);
        void unify(ast::ASTNode *t1, ast::ASTNode *const t2);
        void unify(const ast::Expression &t1, const ast::Statement &t2);

        std::unordered_map<const ast::ASTNode *, type_id_t> typeIDs;
        void unify(const std::vector<ast::Expression> &t1, const ast::Expression &t2);
        void unify(const std::vector<ast::ASTNode *> &t1, const ast::ASTNode *t2);
        void unify(const std::vector<ast::Expression> &t1, const ast::Statement &t2);
    };
} // namespace voila
