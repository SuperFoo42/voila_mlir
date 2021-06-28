#pragma once
#include "DotVisualizer.hpp"
#include "JITInvocationError.hpp"
#include "LLVMGenerationError.hpp"
#include "LLVMOptimizationError.hpp"
#include "MLIRGenerationError.hpp"
#include "MLIRGenerator.hpp"
#include "MLIRLoweringError.hpp"
#include "TypeInferer.hpp"
#include "ast/Fun.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/ShapeInferencePass.hpp>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/VoilaDialect.h>
#include <mlir/lowering/VoilaToAffineLoweringPass.hpp>
#include <mlir/lowering/VoilaToLLVMLoweringPass.hpp>
#pragma GCC diagnostic pop

namespace voila
{
    class Parameter
    {
        [[maybe_unused]] void *data;
        [[maybe_unused]] size_t size;
        [[maybe_unused]] DataType type;
        Parameter(void *data, size_t size, DataType type) : data{data}, size{size}, type{type} {}
      public:
        friend Parameter make_param(void *, size_t , DataType );
        friend class Program;
    };

    class Program
    {
        std::unordered_map<std::string, ast::Expression> func_vars;
        std::vector<void *> params;
        ::mlir::MLIRContext context;
        llvm::LLVMContext llvmContext;
        ::mlir::OwningModuleRef mlirModule;
        std::unique_ptr<llvm::Module> llvmModule;

      public:
        const ::mlir::MLIRContext &getMLIRContext() const;

        const ::mlir::OwningModuleRef &getMLIRModule() const;
        std::vector<std::unique_ptr<ast::Fun>> functions;
        TypeInferer inferer;
        const bool debug;
        const bool m_optimize;

        explicit Program(bool debug = false, bool optimize = true) :
            func_vars(),
            context(),
            llvmContext(),
            mlirModule(),
            llvmModule(),
            functions(),
            inferer(),
            debug{debug},
            m_optimize{optimize}
        {
        }

        ~Program() = default;

        void add_func(ast::Fun *f);

        std::vector<std::unique_ptr<ast::Fun>> &get_funcs()
        {
            return functions;
        }

        ::mlir::OwningModuleRef &generateMLIR();

        void lowerMLIR(bool optimize);

        void convertToLLVM(bool optimize);

        /**
         * @Deprecated use () instead
         * @param shapes
         */
        void runJIT(bool optimize, std::optional<std::string> objPath);

        void printMLIR(const std::string &filename);

        void printLLVM(const std::string &filename);

        void infer_type(const ast::ASTNode &node);
        void infer_type(const ast::Expression &node);
        void infer_type(const ast::Statement &node);

        void to_dot(const std::string &);

        void add_var(ast::Expression expr);

        ast::Expression get_var(const std::string &var_name);

        bool has_var(const std::string &var_name);
        /**
         * @Deprecated use << instead
         * @param shapes
         */
        void set_main_args_shape(const std::unordered_map<std::string, size_t> &shapes);
        /**
         * @Deprecated use << instead
         * @param shapes
         */
        void set_main_args_type(const std::unordered_map<std::string, DataType> &types);

        Program &operator<<(Parameter param);

        std::unique_ptr<void *> operator()();
    };

    Parameter make_param(void *, size_t , DataType );
} // namespace voila
