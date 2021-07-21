#include "Tables.hpp"
#include "Config.hpp"
#include "Program.hpp"
#include "benchmark_defs.hpp.inc"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
static void BM_TPC_Q6(benchmark::State &state)
{
    using namespace voila;

    auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
    auto l_quantity = lineitem.getColumn<int64_t>(4);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_shipdate = lineitem.getColumn<int64_t>(10);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    Config config;
    config.debug = true;
    config.optimize = true;
    const auto file = VOILA_BENCHMARK_SOURCES_PATH "/Q6.voila";
    for ([[maybe_unused]] auto _ : state)
    {
        Program prog(file, config);
        prog << ::voila::make_param(l_quantity.data(), l_quantity.size(), DataType::INT64);
        prog << ::voila::make_param(l_discount.data(), l_discount.size(), DataType::DBL);
        prog << ::voila::make_param(l_shipdate.data(), l_shipdate.size(), DataType::INT64);
        prog << ::voila::make_param(l_extendedprice.data(), l_extendedprice.size(), DataType::DBL);

        // run in jit
        auto res = prog();

        //std::cout << res << std::endl;
    }
}
BENCHMARK(BM_TPC_Q6);