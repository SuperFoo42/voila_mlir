#include "voila_lexer.hpp"
#include "voila_parser.hpp"

#include <cstdlib>
#include <cxxopts.hpp>

void invoke_parser(const std::string &file)
{
    FILE *f = std::fopen(file.c_str(), "r");
    if (f != nullptr)
    {
        voila::lexer::Lexer lexer(f); // read file, decode UTF-8/16/32 format
        lexer.filename = file;        // the filename to display with error locations
        voila::parser::Parser parser(lexer);
        if (parser.parse() != 0)
            std::cerr << "error parsing file" << std::endl;
    }
    else
    {
        std::cerr << "error opening file" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("VOILA compiler", "");

    options.add_options()("h, help", "Show help")(
        "f,file", "File name", cxxopts::value<std::string>()) ("a, plot-ast", "Generate dot file of AST",
                                                               cxxopts::value<bool>()->default_value("false"));

    try
    {
        auto cmd = options.parse(argc, argv);

        if (cmd.count("h"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(EXIT_SUCCESS);
        }

        invoke_parser(cmd["f"].as<std::string>());
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}