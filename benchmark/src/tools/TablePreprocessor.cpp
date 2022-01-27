#include "ColumnTypes.hpp"
#include "TableReader.hpp"
#include "Tables.hpp"

#include <cxxopts.hpp>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    cxxopts::Options options("VOILA benchmark preprocessing tool",
                             "This tool prepares csv files for use in the benchmarks. Therefore, it dictionary "
                             "compresses string columns and converts date columns to int");
    options.add_options()("h, help", "Show help")("f, file", "File name", cxxopts::value<std::vector<std::string>>())(
        "d, delim", "Delimiter used in csv file", cxxopts::value<char>()->default_value("|"))(
        "t, types", "Column types in csv file", cxxopts::value<std::vector<ColumnTypes>>())(
        "o, output", "Filename to write preprocessed table", cxxopts::value<std::string>())(
        "j, join", "Join Attributes", cxxopts::value<std::vector<int>>())("s, sizes", "Table Sizes",
                                                                          cxxopts::value<std::vector<int>>());

    try
    {
        auto cmd = options.parse(argc, argv);
        if (cmd.count("h"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(EXIT_SUCCESS);
        }

        if (cmd.count("f") == 1)
        {
            CompressingTableReader reader(cmd["f"].as<std::vector<std::string>>().front(),
                                                cmd["t"].as<std::vector<ColumnTypes>>(), cmd["d"].as<char>());
            auto compressedTable = reader.getTable();
            compressedTable.writeTable(cmd["o"].as<std::string>());
        }
        else if (cmd.count("f") > 1)
        {
            const auto files = cmd["f"].as<std::vector<std::string>>();
            const auto tps = cmd["t"].as<std::vector<ColumnTypes>>();
            auto joinCols = cmd["j"].as<std::vector<int>>();
            const auto sizes = cmd["s"].as<std::vector<int>>();
            std::reverse(joinCols.begin(), joinCols.end());
            assert(files.size() == sizes.size() && tps.size() == static_cast<size_t>(std::accumulate(sizes.begin(), sizes.end(), 0)) &&
                   "expect type specifications match the number of input tables");
            assert(joinCols.size() == 2 * (files.size() - 1));
            const auto delim = cmd["d"].as<char>();
            std::vector<int> bounds;
            std::inclusive_scan(sizes.begin(), sizes.end(), std::back_inserter(bounds));
            bounds.push_back(std::accumulate(sizes.begin(), sizes.end(), 0));
            auto t = std::vector(tps.begin(), std::next(tps.begin(), bounds.front()));
            Table wideTable = TableReader(files.at(0), t, delim).getTable();
            for (size_t i = 1; i < files.size(); ++i)
            {
                t = std::vector(std::next(tps.begin(), bounds[i - 1]), std::next(tps.begin(), bounds[i]));
                TableReader reader(files.at(i), t, delim);
                auto tmp = reader.getTable();
                const auto jc1 = joinCols.back();
                joinCols.pop_back();
                const auto jc2 = joinCols.back();
                joinCols.pop_back();
                wideTable = wideTable.join(tmp, jc1, jc2);
            }

            CompressedTable(std::move(wideTable)).writeTable(cmd["o"].as<std::string>());
        }
        else
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }
}
