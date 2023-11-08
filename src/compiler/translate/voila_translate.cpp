
// DEPRECATED: functionality is integrated into voila-opt. This tool will be removed soon
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "Program.hpp"                                  // for Program, mak...
#include "Types.hpp"                                    // for Arity, DataT...
#include "mlir/Dialects/Voila/IR/VoilaOpsDialect.h.inc" // for VoilaDialect
#include "mlir/IR/DialectRegistry.h"                    // for DialectRegistry
#include "llvm/ADT/SmallVector.h"                       // for SmallVector
#include "llvm/ADT/StringRef.h"                         // for StringRef
#include <charconv>                                     // for from_chars
#include <cstdlib>                                      // for size_t, EXIT...
#include <filesystem>                                   // for is_regular_file
#include <llvm/Support/CommandLine.h>                   // for opt, desc
#include <magic_enum.hpp>                               // for enum_cast
#include <mlir/IR/MLIRContext.h>                        // for registerMLIR...
#include <mlir/InitAllDialects.h>                       // for registerAllD...
#include <mlir/InitAllPasses.h>                         // for registerAllP...
#include <optional>                                     // for optional
#include <stdexcept>                                    // for invalid_argu...
#include <string>                                       // for string, oper...
#include <system_error>                                 // for errc
#include <vector>                                       // for vector
#pragma GCC diagnostic pop
/*#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <magic_enum.hpp>
#include <mlir/Dialects/Voila/IR/VoilaDialect.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#pragma GCC diagnostic pop

#include <charconv>*/
#include <cstdlib>
#include <filesystem>

namespace cl = llvm::cl;

namespace
{
    enum Action
    {
        None,
        DumpAST,
        DumpMLIR
    };

    struct VType
    {
        voila::DataType type;
        voila::Arity ar;
    };
} // namespace

namespace voila
{
    template <> Parameter make_param(VType &val) { return make_param(nullptr, val.ar.get_size(), val.type); }
    template <> Parameter make_param(VType *val) { return make_param(nullptr, val->ar.get_size(), val->type); }
} // namespace voila

namespace llvm::cl
{
    template <> class cl::parser<VType> : public basic_parser<VType>
    {
      public:
        using parser_data_type = VType;

        explicit parser(Option &O) : basic_parser(O) {}

        bool parse(cl::Option &O, StringRef, const StringRef ArgValue, VType &Val)
        {
            std::string::size_type n = ArgValue.find(':');
            auto maybeType = magic_enum::enum_cast<voila::DataType>(ArgValue.substr(0, n));
            if (maybeType.has_value())
            {
                Val.type = maybeType.value();
            }
            else
            {
                O.error("TODO");
                return true;
            }

            if (n != llvm::StringRef::npos)
            {
                auto as = ArgValue.substr(n + 1);
                size_t result{};

                auto [ptr, ec]{std::from_chars(as.data(), as.data() + as.size(), result)};

                if (ec == std::errc())
                {
                    Val.ar = voila::Arity(result);
                }
                else if (ec == std::errc::invalid_argument)
                {
                    O.error("That isn't a number.");
                    return true;
                }
                else if (ec == std::errc::result_out_of_range)
                {
                    O.error("This number is larger than an size_t.");
                    return true;
                }
            }
            else
            {
                Val.ar = voila::Arity(0);
            }

            return false;
        }

        [[nodiscard]] StringRef getValueName() const override { return "Voila Type"; }
    };
} // namespace llvm::cl

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input voila file>"), cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
    outputFilename("o", cl::desc("output mlir filename"), cl::init("-"), cl::value_desc("filename"));

cl::list<VType> paramTypes(cl::Positional, cl::desc("<parameter types in form TYPE:Cardinality>"));

static cl::opt<enum Action> emitAction("emit",
                                       cl::desc("Select the kind of output desired"),
                                       cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
                                       cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

int main(int argc, char *argv[])
{
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::voila::VoilaDialect>();
    registerAllDialects(registry);
    mlir::registerMLIRContextCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "voila translate\n");

    if (inputFilename != "-" && !std::filesystem::is_regular_file(std::filesystem::path(inputFilename.data())))
    {
        throw std::invalid_argument("invalid input file");
    }

    if (outputFilename != "-" && !std::filesystem::is_regular_file(std::filesystem::path(outputFilename.data())))
    {
        throw std::invalid_argument("invalid output file");
    }

    auto prog = voila::Program(voila::InputType::Voila,inputFilename, paramTypes.begin(), paramTypes.end());

    switch (emitAction)
    {
    case None:
        break;
    case DumpAST:
        prog.to_dot(outputFilename);
        break;
    case DumpMLIR:
        prog.generateMLIR();
        prog.printMLIR(outputFilename);
        break;
    }

    return EXIT_SUCCESS;
}