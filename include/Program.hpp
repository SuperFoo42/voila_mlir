#pragma once

#include <memory>
#include <range/v3/all.hpp>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <filesystem>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "llvm/IR/LLVMContext.h"
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/InitAllDialects.h>
#include <mlir/IR/BuiltinOps.h>
#pragma GCC diagnostic pop

#include "TypeInferer.hpp"
#include "Types.hpp"
#include "ast/Expression.hpp"
#include "Config.hpp"

namespace mlir
{
    class ExecutionEngine;
}

namespace voila
{
    namespace ast
    {
        class Expression;
        class Fun;
        class Statement;
    } // namespace ast
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
        ::mlir::OwningOpRef<::mlir::ModuleOp> mlirModule;
        std::unique_ptr<llvm::Module> llvmModule;
        std::optional<std::unique_ptr<::mlir::ExecutionEngine>> maybeEngine;
        std::unordered_map<std::string, std::shared_ptr<ast::Fun>> functions;
        Config config;
        std::unique_ptr<lexer::Lexer> lexer;
        long long timer = 0;
        int64_t max_in_table_size;
        void runJIT(const std::optional<std::string> &objPath = std::nullopt);

      public:
        using result_t = std::variant<strided_memref_ptr<uint32_t, 1>,
                                      strided_memref_ptr<uint64_t, 1>,
                                      strided_memref_ptr<double, 1>,
                                      uint32_t,
                                      uint64_t,
                                      double>;

        [[maybe_unused]] const ::mlir::MLIRContext &getMLIRContext() const;

        [[maybe_unused]] const ::mlir::OwningOpRef<::mlir::ModuleOp> &getMLIRModule() const;

        explicit Program(Config config = Config());

        explicit Program(const std::string &source_path  = "-", Config config = Config());

        ~Program();

        void add_func(ast::Fun *f);

        void add_func(std::shared_ptr<ast::Fun> f) {
            f->variables() = std::move(func_vars);
            functions.emplace(f->name() + "_" + to_string(*inferer.get_type(*f)), std::move(f));
            func_vars.clear();
        }

        auto get_funcs() const
        {
            return ranges::views::values(functions);
        }

        const auto &get_func(const std::string &fName, const FunctionType &type)
        {
            return functions.at(fName + "_" + to_string(type));
        }

        auto has_func(const std::string &fName, const FunctionType &type)
        {
            return functions.contains(fName + "_" + to_string(type));
        }

        const auto &get_func(const std::string &func_name) const
        {
            return functions.at(func_name);
        }

        ::mlir::OwningOpRef<::mlir::ModuleOp> &generateMLIR();

        void lowerMLIR();

        void convertToLLVM();

        std::unique_ptr<::mlir::ExecutionEngine> &getOrCreateExecutionEngine();

        /**
         * @Deprecated use () instead
         * @param shapes
         */

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

        double getExecTime() const
        {
            return timer;
        }

        TypeInferer inferer;
    };

    Parameter make_param(void *, size_t, DataType);

} // namespace voila
