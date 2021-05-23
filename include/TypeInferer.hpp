#pragma once
#include "ASTNodes.hpp"
#include "Type.hpp"

#include <ast/ASTVisitor.hpp>
#include <unordered_map>

namespace voila
{
    class TypeInferer : public ast::ASTVisitor
    {
        std::unordered_map<const ast::ASTNode *, size_t> typeIDs;
        std::vector<std::unique_ptr<Type>> types;

      public:
        //TypeInferer() : typeIDs{}, types{} {}

        void operator()(const ast::Aggregation &aggregation) final;
        void operator()(const ast::Write &write) final;
        void operator()(const ast::Scatter &scatter) final;
        void operator()(const ast::FunctionCall &call) final;
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

        Type &get_type(const ast::ASTNode &node) const;

        Type &get_type(const ast::Expression &node) const;

        Type &get_type(const ast::Statement &node) const;

      private:
        void unify(const ast::ASTNode &t1, const ast::ASTNode &t2);
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

        void insertNewType(const ast::ASTNode &node, DataType t, Arity ar);
        void
        insertNewFuncType(const ast::ASTNode &node, std::vector<size_t> typeParamIDs, DataType returnT, Arity returnAr);

        size_t get_type_id(const ast::Expression &node);
        size_t get_type_id(const ast::Statement &node);
        size_t get_type_id(const ast::ASTNode &node);
        void unify(const ast::ASTNode &t1, const ast::Statement &t2);
        void unify(const ast::ASTNode &t1, const ast::Expression &t2);
        void insertNewFuncType(const ast::ASTNode &node, std::vector<size_t> typeParamIDs, const size_t returnTypeID);
    };
} // namespace voila
