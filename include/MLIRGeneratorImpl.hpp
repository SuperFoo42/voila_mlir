#pragma once
#include "ASTNodes.hpp"
#include "Program.hpp"

#include <NotInferedException.hpp>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <variant>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "VariableAlreadyDeclaredException.hpp"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#pragma GCC diagnostic pop
namespace voila::mlir
{
    class MLIRGeneratorImpl : public ast::ASTVisitor
    {
        ::mlir::OpBuilder &builder;
        ::mlir::ModuleOp &module;
        llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable;
        std::unordered_map<std::string, ::mlir::FuncOp> &funcTable;
        const TypeInferer &inferer;
        using result_variant = std::variant<std::monostate,
                                            ::mlir::ModuleOp,
                                            ::mlir::Value,
                                            ::mlir::ValueRange,
                                            ::mlir::Type,
                                            ::mlir::LogicalResult,
                                            ::mlir::FuncOp>;
        result_variant result;

        // helper functions to map ast types to mlir
        ::mlir::Location loc(ast::Location loc);

        ::mlir::Type getType(const ast::ASTNode &node);

        // TODO: implement
        ::mlir::Type convert(const Type &t);

        // TODO: is this correct?
        void declare(llvm::StringRef var, ::mlir::Value value)
        {
            (void) module;
            if (symbolTable.count(var))
                throw ast::VariableAlreadyDeclaredException();
            symbolTable.insert(var, value);
        }

        void mlirGenBody(const std::vector<ast::Statement> &block);

        result_variant visitor_gen(const ast::Statement &node);

        result_variant visitor_gen(const ast::Expression &node);

        static llvm::ArrayRef<int64_t> getShape(const ::mlir::Value &lhs, const ::mlir::Value &rhs)
        {
            llvm::ArrayRef<int64_t> shape;
            if (lhs.getType().isa<::mlir::TensorType>() &&
                lhs.getType().dyn_cast<::mlir::RankedTensorType>().hasStaticShape())
            {
                shape = lhs.getType().dyn_cast<::mlir::RankedTensorType>().getShape();
            }
            else if (rhs.getType().isa<::mlir::TensorType>() &&
                     rhs.getType().dyn_cast<::mlir::RankedTensorType>().hasStaticShape())
            {
                shape = rhs.getType().dyn_cast<::mlir::RankedTensorType>().getShape();
            }
            else
            {
                shape = llvm::SmallVector<int64_t, 1>{-1};
            }
            return shape;
        }

        template<class Op>
        ::mlir::Value getCmpOp(const ast::Comparison &cmpNode)
        {
            auto location = loc(cmpNode.get_location());
            auto lhs = std::get<::mlir::Value>(visitor_gen(cmpNode.lhs));
            auto rhs = std::get<::mlir::Value>(visitor_gen(cmpNode.rhs));
            if (lhs.getType().isa<::mlir::TensorType>() || rhs.getType().isa<::mlir::TensorType>())
            {
                ::mlir::ArrayRef<int64_t> shape;
                shape = getShape(lhs, rhs);

                return builder.create<Op>(location, ::mlir::RankedTensorType::get(shape, builder.getI1Type()), lhs,
                                          rhs);
            }
            else
                return builder.create<Op>(location, builder.getI1Type(), lhs, rhs);
        }

        ::mlir::Type getScalarType(const ast::ASTNode &node);

        ::mlir::Type scalarConvert(const ::voila::Type &t);

      public:
        MLIRGeneratorImpl(::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp &module,
                          ::llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable,
                          std::unordered_map<std::string, ::mlir::FuncOp> &funcTable,
                          const TypeInferer &inferer) :
            builder{builder}, module{module}, symbolTable{symbolTable}, funcTable{funcTable}, inferer{inferer}, result{}
        {
            (void) module;
        }

        result_variant getValue()
        {
            return result;
        }

        void operator()(const ast::AggrSum &sum) final;
        void operator()(const ast::AggrCnt &cnt) final;
        void operator()(const ast::AggrMin &min) final;
        void operator()(const ast::AggrMax &max) final;
        void operator()(const ast::AggrAvg &avg) final;
        void operator()(const ast::Write &write) final;
        void operator()(const ast::Scatter &scatter) final;
        void operator()(const ast::FunctionCall &call) final;
        void operator()(const ast::Assign &assign) final;
        void operator()(const ast::Emit &emit) final;
        void operator()(const ast::Loop &loop) final;
        void operator()(const ast::StatementWrapper &wrapper) final;
        void operator()(const ast::Add &add) final;
        void operator()(const ast::Sub &sub) final;
        void operator()(const ast::Mul &mul) final;
        void operator()(const ast::Div &div) final;
        void operator()(const ast::Mod &mod) final;
        void operator()(const ast::Eq &eq) final;
        void operator()(const ast::Neq &neq) final;
        void operator()(const ast::Le &le) final;
        void operator()(const ast::Ge &ge) final;
        void operator()(const ast::Leq &leq) final;
        void operator()(const ast::Geq &geq) final;
        void operator()(const ast::And &anAnd) final;
        void operator()(const ast::Or &anOr) final;
        void operator()(const ast::Not &aNot) final;
        void operator()(const ast::IntConst &intConst) final;
        void operator()(const ast::BooleanConst &booleanConst) final;
        void operator()(const ast::FltConst &fltConst) final;
        void operator()(const ast::StrConst &aConst) final;
        void operator()(const ast::Read &read) final;
        void operator()(const ast::Gather &gather) final;
        void operator()(const ast::Ref &param) final;
        void operator()(const ast::TupleGet &get) final;
        void operator()(const ast::TupleCreate &create) final;
        void operator()(const ast::Fun &fun) final;
        void operator()(const ast::Main &main) final;
        void operator()(const ast::Selection &selection) final;
        void operator()(const ast::Variable &variable) final;
        void operator()(const ast::Predicate &pred) final;
        void operator()(const ast::Hash &hash) final;
        void operator()(const ast::Lookup &lookup) override;
        void operator()(const ast::Insert &insert) override;
    };
} // namespace voila::mlir
