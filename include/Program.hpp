#pragma once

#include "Config.hpp"      // for Config
#include "TypeInferer.hpp" // for TypeInferer
#include "Types.hpp"       // for to_string, DataType
#include "ast/Fun.hpp"     // for Fun
#include <cassert>         // for assert
#include <cstdint>         // for int64_t, uint32_t
#include <cstdlib>         // for free, size_t
#include <fstream>
#include <memory>        // for allocator, shared_ptr
#include <optional>      // for optional, nullopt
#include <string>        // for string, operator+
#include <unordered_map> // for operator==, unorde...
#include <utility>       // for move
#include <variant>       // for variant
#include <vector>        // for vector
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/ExecutionEngine/ExecutionEngine.h" // for ExecutionEngine
#include "llvm/ADT/SmallVector.h"                 // for SmallVector
#include "llvm/IR/LLVMContext.h"                  // for LLVMContext
#include <mlir/IR/BuiltinOps.h>                   // for ModuleOp
#include <mlir/IR/MLIRContext.h>                  // for MLIRContext
#include <mlir/IR/OwningOpRef.h>                  // for OwningOpRef
#pragma GCC diagnostic pop

#include "ParsingError.hpp"
#include "range/v3/view/all.hpp"    // for all_t
#include "range/v3/view/filter.hpp" // for filter_view, cpp20...
#include "range/v3/view/map.hpp"    // for values, values_fn
#include "range/v3/view/view.hpp"   // for view_closure
// #include "voila_parser.hpp"
#include "voila_lexer.hpp"
template <typename T, int N> struct StridedMemRefType;

namespace llvm
{
    class Module;
}
namespace voila
{
    enum InputType
    {
        Voila,
        MLIR
    };

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

    template <class T, int N> using strided_memref_ptr = std::shared_ptr<StridedMemRefType<T, N>>;

    template <typename T> struct TypeMap
    {
        static_assert(false && "Could not deduce type of parameter");
    };

    template <> struct TypeMap<int32_t>
    {
        const static auto type = DataType::INT32;
    };

    template <> struct TypeMap<int64_t>
    {
        const static auto type = DataType::INT64;
    };

    template <> struct TypeMap<double>
    {
        const static auto type = DataType::DBL;
    };

    template <> struct TypeMap<bool>
    {
        const static auto type = DataType::BOOL;
    };

    template <typename T> Parameter make_param(T *val, size_t size) { return make_param(val, size, TypeMap<T>::type); }

    template <typename T> Parameter make_param(std::vector<T> &val) { return make_param(val.data(), val.size()); }

    template <typename T> Parameter make_param(T *val) { return make_param(val, 0); }
    template <typename T> Parameter make_param(T &val) { return make_param(&val, 0); }

    class Program
    {
        std::unordered_map<std::string, ast::ASTNodeVariant> func_vars;
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
        void loadInputFile(InputType type, const std::string &source_path);

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

        template <class... Ts>
        explicit Program(const InputType inputType,
                         const std::string &source_path = "-",
                         Ts... params,
                         Config config = Config())
            : func_vars(),
              context(),
              llvmContext(),
              mlirModule(),
              llvmModule(),
              maybeEngine(std::nullopt),
              functions(),
              config{std::move(config)},
              max_in_table_size(0),
              inferer(this)
        {
            loadInputFile(inputType, source_path);

            // supply program parameters
            (this->operator<<(std::forward<Ts>(params)), ...);

            this->inferTypes();
        }

        template <class Iterator>
        explicit Program(const InputType inputType,
                         const std::string &source_path,
                         Iterator begin,
                         Iterator end,
                         Config config = Config())
            : func_vars(),
              context(),
              llvmContext(),
              mlirModule(),
              llvmModule(),
              maybeEngine(std::nullopt),
              functions(),
              config{std::move(config)},
              max_in_table_size(0),
              inferer(this)
        {
            loadInputFile(inputType, source_path);

            // supply program parameters
            std::for_each(begin, end, [this](auto &el) { (*this) << el; });

            inferTypes();
        }

        ~Program();

        void add_func(ast::ASTNodeVariant f);
        void add_cloned_func(ast::ASTNodeVariant f);

        auto funcs() const { return ranges::views::values(functions); }

        auto life_funcs() const
        {
            return ranges::views::values(ranges::views::filter(
                functions, [](auto &en) { return en.first == "main" || en.first.ends_with("_ret_"); }));
        }

        const auto &get_func(const std::string &fName, const FunctionType &type)
        {
            return functions.at(fName + "_" + to_string(type));
        }

        auto has_func(const std::string &fName, const FunctionType &type)
        {
            return functions.contains(fName + "_" + to_string(type));
        }

        const auto &get_func(const std::string &func_name) const { return functions.at(func_name); }

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

        void inferTypes();

        void to_dot(const std::string &);

        void add_var(const ast::ASTNodeVariant &expr);

        ast::ASTNodeVariant get_var(const std::string &var_name);

        bool has_var(const std::string &var_name);

        Program &operator<<(Parameter param);

        template <class T>
        Program &operator<<(T param)
            requires std::is_pointer_v<T>
        {
            return *this << make_param(param);
        }

        template <class T> Program &operator<<(T &param) { return *this << make_param(param); }

        std::vector<result_t> operator()();

        double getExecTime() const { return timer; }

        TypeInferer inferer;
    };

    Parameter make_param(void *, size_t, DataType);

} // namespace voila
