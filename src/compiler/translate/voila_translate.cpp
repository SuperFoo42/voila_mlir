#include "Config.hpp"
#include "ParsingError.hpp"
#include "Program.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Dialects/Voila/IR/VoilaDialect.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Support/FileUtilities.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>

#pragma GCC diagnostic pop

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <charconv>

namespace cl = llvm::cl;

namespace {
    enum Action {
        None, DumpAST, DumpMLIR
    };

    struct VType {
        voila::DataType type;
        voila::Arity ar;
    };
} // namespace

namespace voila {
    template<>
    Parameter make_param(VType *val) {
        return make_param(nullptr, val->ar.get_size(), val->type);
    }
}

namespace llvm::cl {
    template<>
    class cl::parser<VType> : public basic_parser<VType> {
    public:
        using parser_data_type = VType;

        explicit parser(Option &O) : basic_parser(O) {}

        bool parse(cl::Option &O, StringRef, const StringRef ArgValue,
                   VType &Val) {
            std::string::size_type n = ArgValue.find(':');
            auto maybeType = magic_enum::enum_cast<voila::DataType>(ArgValue.substr(0, n));
            if (maybeType.has_value()) {
                Val.type = maybeType.value();
            } else {
                O.error("TODO");
                return true;
            }

            if (n != std::string::npos) {
                auto as = ArgValue.substr(n + 1);
                size_t result{};

                auto [ptr, ec] {std::from_chars(as.data(), as.data() + as.size(), result)};

                if (ec == std::errc()) {
                    Val.ar = voila::Arity(result);
                } else if (ec == std::errc::invalid_argument) {
                    O.error("That isn't a number.");
                    return true;
                } else if (ec == std::errc::result_out_of_range) {
                    O.error("This number is larger than an size_t.");
                    return true;
                }

            }

            return false;
        }

        [[nodiscard]] StringRef getValueName() const override { return "Voila Type"; }
    };
}

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input voila file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("output mlir filename"), cl::init("-"),
                                           cl::value_desc("filename"));

cl::list<VType> paramTypes(cl::Positional, cl::desc("<parameter types in form TYPE:Cardinality>"));

static cl::opt<enum Action> emitAction("emit", cl::desc("Select the kind of output desired"),
                                       cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")
                                       ),
                                       cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")
                                       ));


int main(int argc, char *argv[]) {
    spdlog::set_level(spdlog::level::warn);
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::voila::VoilaDialect>();
    registerAllDialects(registry);
    mlir::registerMLIRContextCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "voila translate\n");

    if (inputFilename != "-" && !std::filesystem::is_regular_file(std::filesystem::path(inputFilename.data()))) {
        throw std::invalid_argument("invalid input file");
    }

    if (outputFilename != "-" && !std::filesystem::is_regular_file(std::filesystem::path(outputFilename.data()))) {
        throw std::invalid_argument("invalid output file");
    }

    auto prog = voila::Program(inputFilename);

    for (auto param: paramTypes)
        prog << &param;

    prog.inferTypes();

    switch (emitAction) {
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