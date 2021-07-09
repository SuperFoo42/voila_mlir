#pragma once
#include "ArgsOutOfRangeError.hpp"
#include "DotVisualizer.hpp"
#include "JITInvocationError.hpp"
#include "LLVMGenerationError.hpp"
#include "LLVMOptimizationError.hpp"
#include "MLIRGenerationError.hpp"
#include "MLIRGenerator.hpp"
#include "MLIRLoweringError.hpp"
#include "NotImplementedException.hpp"
#include "ParsingError.hpp"
#include "TypeInferencePass.hpp"
#include "TypeInferer.hpp"
#include "ast/Fun.hpp"

#include <fstream>
#include <llvm/IR/AssemblyAnnotationWriter.h>
#include <memory>
#include <spdlog/spdlog.h>
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
#include <mlir/ExecutionEngine/MemRefUtils.h>
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
#include <range/v3/all.hpp>
#include <unordered_map>
#pragma GCC diagnostic pop

namespace voila
{
    namespace lexer
    {
        class Lexer;
    }
    class Parameter
    {
        void *data;
        size_t size;
        DataType type;
        Parameter(void *data, size_t size, DataType type) : data{data}, size{size}, type{type} {}

      public:
        friend Parameter make_param(void *, size_t, DataType);
        friend class Program;
    };

    class Program
    {
        std::unordered_map<std::string, ast::Expression> func_vars;
        ::mlir::SmallVector<void *, 11> params; // 11 is two memrefs + result
        size_t nparam = 0;
        ::mlir::MLIRContext context;
        llvm::LLVMContext llvmContext;
        ::mlir::OwningModuleRef mlirModule;
        std::unique_ptr<llvm::Module> llvmModule;
        std::unordered_map<std::string, std::unique_ptr<ast::Fun>> functions;
        const bool debug;
        const bool m_optimize;
        lexer::Lexer *lexer;

      public:
        const ::mlir::MLIRContext &getMLIRContext() const;

        const ::mlir::OwningModuleRef &getMLIRModule() const;

        explicit Program(bool debug = false, bool optimize = true);

        explicit Program(std::string_view source_path, bool debug = false, bool optimize = true);

        ~Program() = default;

        void add_func(ast::Fun *f);

        auto get_funcs() const
        {
            return ranges::views::values(functions);
        }

        ::mlir::OwningModuleRef &generateMLIR();

        void lowerMLIR();

        void convertToLLVM();

        /**
         * @Deprecated use () instead
         * @param shapes
         */
        void runJIT(std::optional<std::string> objPath = std::nullopt);

        void printMLIR(const std::string &filename);

        void printLLVM(const std::string &filename);

        void infer_type(const ast::ASTNode &node);
        void infer_type(const ast::Expression &node);
        void infer_type(const ast::Statement &node);

        void inferTypes();

        void to_dot(const std::string &);

        void add_var(ast::Expression expr);

        ast::Expression get_var(const std::string &var_name);

        bool has_var(const std::string &var_name);

        Program &operator<<(Parameter param);

        std::variant<std::monostate,
                     std::unique_ptr<StridedMemRefType<uint32_t, 1> *>,
                     std::unique_ptr<StridedMemRefType<uint64_t, 1> *>,
                     std::unique_ptr<StridedMemRefType<double, 1> *>,
                     uint32_t,
                     uint64_t,
                     double>
        operator()();

        TypeInferer inferer;
    };

    Parameter make_param(void *, size_t, DataType);
} // namespace voila
