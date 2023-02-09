#pragma once

#include <cassert>                                // for assert
#include <cstdint>                                // for int64_t, uint32_t
#include <cstdlib>                                 // for free, size_t
#include <memory>                                  // for allocator, shared_ptr
#include <optional>                                // for optional, nullopt
#include <string>                                  // for string, operator+
#include <unordered_map>                           // for operator==, unorde...
#include <utility>                                 // for move
#include <variant>                                 // for variant
#include <vector>                                  // for vector
#include "Config.hpp"                              // for Config
#include "TypeInferer.hpp"                         // for TypeInferer
#include "Types.hpp"                               // for to_string, DataType
#include "ast/Expression.hpp"                      // for Expression
#include "ast/Fun.hpp"                             // for Fun

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/BuiltinOps.h>                    // for ModuleOp
#include <mlir/IR/MLIRContext.h>                   // for MLIRContext
#include <mlir/IR/OwningOpRef.h>                   // for OwningOpRef
#include "llvm/ADT/SmallVector.h"                  // for SmallVector
#include "llvm/IR/LLVMContext.h"                   // for LLVMContext
#include "mlir/ExecutionEngine/ExecutionEngine.h"  // for ExecutionEngine
//#include "mlir/ExecutionEngine/CRunnerUtils.h"      // for StridedMemref
#pragma GCC diagnostic pop

#include "range/v3/view/all.hpp"                   // for all_t
#include "range/v3/view/filter.hpp"                // for filter_view, cpp20...
#include "range/v3/view/map.hpp"                   // for values, values_fn
#include "range/v3/view/view.hpp"                  // for view_closure

template <typename T, int N> struct StridedMemRefType;

namespace llvm { class Module; }
namespace voila
{
    namespace ast
    {
        class Statement;
        class ASTNode;
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

        auto funcs() const
        {
            return ranges::views::values(functions);
        }

        auto life_funcs() const
        {
            return ranges::views::values(ranges::views::filter(functions, [](auto &en) { return en.first == "main" ||
                                                                                                en.first.ends_with(
                                                                                                        "_ret_");}));
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
