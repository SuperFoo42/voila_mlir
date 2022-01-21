#pragma once
#include "ArgsOutOfRangeError.hpp"
#include "Config.hpp"
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
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/VoilaDialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Passes/VoilaToAffineLoweringPass.hpp>
#include <mlir/Passes/VoilaToLLVMLoweringPass.hpp>
#include <mlir/Passes/VoilaToLinalgLoweringPass.hpp>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/lowering/LinalgTiledLoopsToAffineForPass.hpp>
#pragma GCC diagnostic pop
#include "Profiler.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <unordered_map>

namespace voila
{
    namespace lexer
    {
        class Lexer;
    }
    class Parameter
    {
        void *data;
        int64_t size;
        DataType type;
        Parameter(void *data, size_t size, DataType type) : data{data}, size(size), type{type} {}
        friend Parameter make_param(void *, size_t, DataType);

      public:
        friend class Program;
    };

    struct ProgramResultDeleter
    {
        llvm::SmallVector<void *> toDealloc;

        void operator()(void *b)
        {
            for (auto &ptr : toDealloc)
            {
                std::free(ptr);
                ptr = nullptr;
            }
            std::free(b);
        }
    };

    template<class T, int N>
    using strided_memref_ptr = std::shared_ptr<StridedMemRefType<T, N>>;

    template<typename T>
    Parameter make_param(T *val, size_t size)
    {
        if constexpr (std::is_same_v<int32_t, T>)
        {
            return make_param(val, size, DataType::INT32);
        }
        else if constexpr (std::is_same_v<int64_t, T>)
        {
            return make_param(val, size, DataType::INT64);
        }
        else if constexpr (std::is_same_v<double, T>)
        {
            return make_param(val, size, DataType::DBL);
        }
        else if constexpr (std::is_same_v<bool, T>)
        {
            return make_param(val, size, DataType::BOOL);
        }
        else
        {
            assert(false && "Could not deduce type of parameter");
        }
    }

    template<typename T>
    Parameter make_param(std::vector<T> &val)
    {
        return make_param(val.data(), val.size());
    }

    template<typename T>
    Parameter make_param(T *val)
    {
        return make_param(val, 0);
    }

    class Program
    {
        std::unordered_map<std::string, ast::Expression> func_vars;
        ::mlir::SmallVector<void *, 11> params; // 11 is two memrefs + result
        ::mlir::SmallVector<void *> toDealloc{};
        size_t nparam = 0;
        ::mlir::MLIRContext context;
        llvm::LLVMContext llvmContext;
        ::mlir::OwningModuleRef mlirModule;
        std::unique_ptr<llvm::Module> llvmModule;
        std::optional<std::unique_ptr<::mlir::ExecutionEngine>> maybeEngine;
        std::unordered_map<std::string, std::unique_ptr<ast::Fun>> functions;
        Config config;
        lexer::Lexer *lexer;
        long_long timer = 0;
        int64_t max_in_table_size;

      public:
        using result_t = std::variant<strided_memref_ptr<uint32_t, 1>,
                                      strided_memref_ptr<uint64_t, 1>,
                                      strided_memref_ptr<double, 1>,
                                      uint32_t,
                                      uint64_t,
                                      double>;

        [[maybe_unused]] const ::mlir::MLIRContext &getMLIRContext() const;

        [[maybe_unused]] const ::mlir::OwningModuleRef &getMLIRModule() const;

        explicit Program(Config config = Config());

        explicit Program(std::string_view source_path, Config config = Config());

        ~Program();

        void add_func(ast::Fun *f);

        auto get_funcs() const
        {
            return ranges::views::values(functions);
        }

        ::mlir::OwningModuleRef &generateMLIR();

        void lowerMLIR();

        void convertToLLVM();

        std::unique_ptr<::mlir::ExecutionEngine> &getOrCreateExecutionEngine();

        /**
         * @Deprecated use () instead
         * @param shapes
         */
        void runJIT(const std::optional<std::string> &objPath = std::nullopt);

        void printMLIR(const std::string &filename);

        void printLLVM(const std::string &filename);

        void infer_type(const ast::ASTNode &node);
        void infer_type(const ast::Expression &node);
        void infer_type(const ast::Statement &node);

        void inferTypes();

        void to_dot(const std::string &);

        void add_var(const ast::Expression &expr);

        ast::Expression get_var(const std::string &var_name);

        bool has_var(const std::string &var_name);

        Program &operator<<(Parameter param);

        template<class T>
        Program &operator<<(T param) requires std::is_pointer_v<T>
        {
            return *this << make_param(param);
        }

        template<class T>
        Program &operator<<(T &param)
        {
            return *this << make_param(param);
        }

        std::vector<result_t> operator()();

        double getExecTime()
        {
            return timer;
        }

        TypeInferer inferer;
    };

    Parameter make_param(void *, size_t, DataType);

} // namespace voila
