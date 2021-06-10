#pragma once
#include "DotVisualizer.hpp"
#include "MLIRGenerationError.hpp"
#include "MLIRLoweringError.hpp"
#include "JITInvocationError.hpp"
#include "LLVMGenerationError.hpp"
#include "LLVMOptimizationError.hpp"
#include "MLIRGenerator.hpp"
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
    class Program
    {
        std::unordered_map<std::string, ast::Expression> func_vars;
        ::mlir::OwningModuleRef mlirModule;
        std::unique_ptr<llvm::Module> llvmModule;
        ::mlir::MLIRContext context;

      public:
        const ::mlir::MLIRContext &getMLIRContext() const;

      public:
        const ::mlir::OwningModuleRef &getMLIRModule() const;

      public:
        std::vector<std::unique_ptr<ast::Fun>> functions;
        TypeInferer inferer;

        Program() = default;

        void add_func(ast::Fun *f)
        {
            functions.emplace_back(f);
            f->variables = std::move(func_vars);
            func_vars.clear();
        }

        std::vector<std::unique_ptr<ast::Fun>> &get_funcs()
        {
            return functions;
        }

        ::mlir::OwningModuleRef &generateMLIR();

        void lowerMLIR(bool optimize);

        void convertToLLVM(bool optimize);

        void runJIT(void *args, bool optimize);

        void printMLIR(const std::string &filename);

        void printLLVM(const std::string &filename);

        void infer_type(const ast::ASTNode &node);
        void infer_type(const ast::Expression &node);
        void infer_type(const ast::Statement &node);

        void to_dot(const std::string &);

        void add_var(ast::Expression expr);

        ast::Expression get_var(const std::string &var_name);

        bool has_var(const std::string &var_name);
        void set_main_args_shape(const std::unordered_map<std::string, size_t> &shapes);
        void set_main_args_type(const std::unordered_map<std::string, DataType> &types);
    };
} // namespace voila
