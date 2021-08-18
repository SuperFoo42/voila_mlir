#include "Config.hpp"
#include "Program.hpp"
#include "Tables.hpp"
#include "benchmark_defs.hpp.inc"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop

/*static void BM_TPC_Q1(benchmark::State &state)
{
    using namespace voila;

    auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    Config config;
    config.debug = false;
    config.optimize = true;
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q1.voila";
    for ([[maybe_unused]] auto _ : state)
    {
        Program prog(query, config);
        prog << ::voila::make_param(l_quantity.data(), l_quantity.size(), DataType::INT32);
        prog << ::voila::make_param(l_discount.data(), l_discount.size(), DataType::DBL);
        prog << ::voila::make_param(l_shipdate.data(), l_shipdate.size(), DataType::INT32);
        prog << ::voila::make_param(l_extendedprice.data(), l_extendedprice.size(), DataType::DBL);

        // run in jit
        auto res = prog();

        // FIXME: wrong type, currently int instead of double
        std::cout << std::get<double>(res) << std::endl;
    }
}*/

static void Q6(benchmark::State &state)
{
    using namespace voila;

    auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);
    Config config;
    config.debug = false;
    config.optimize = true;
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q6.voila";

    double queryTime = 0;

    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        prog << ::voila::make_param(l_quantity.data(), l_quantity.size(), DataType::INT32);
        prog << ::voila::make_param(l_discount.data(), l_discount.size(), DataType::DBL);
        prog << ::voila::make_param(l_shipdate.data(), l_shipdate.size(), DataType::INT32);
        prog << ::voila::make_param(l_extendedprice.data(), l_extendedprice.size(), DataType::DBL);

        // run in jit
        auto res = prog();
        queryTime += prog.getExecTime();
        // FIXME: wrong type, currently int instead of double
        // std::cout << std::get<double>(res) << std::endl;
    }
    state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
}

/*static void BM_TPC_Q1Baseline(benchmark::State &state)
{
    using namespace voila;

    auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    for ([[maybe_unused]] auto _ : state)
    {
        double res = 0;
        for (size_t i = 0; i < l_quantity.size(); ++i)
        {
            if (l_shipdate[i] >= 19940101 && l_shipdate[i] < 19950101 && l_quantity[i] < 24 && l_discount[i] >= 0.05 &&
                l_discount[i] <= 0.07)
            {
                res += l_extendedprice[i] * l_discount[i];
            }
        }
        std::cout << res << std::endl;
    }
}*/

static void Q6_Baseline(benchmark::State &state)
{
    using namespace voila;

    auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_tax = lineitem.getColumn<double>(7);
    auto l_returnflag = lineitem.getColumn<int32_t>(8);
    auto l_linestatus = lineitem.getColumn<int32_t>(9);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);

    for ([[maybe_unused]] auto _ : state)
    {
        double res = 0;
        for (size_t i = 0; i < l_quantity.size(); ++i)
        {
            if (l_shipdate[i] >= 19940101 && l_shipdate[i] < 19950101 && l_quantity[i] < 24 && l_discount[i] >= 0.05 &&
                l_discount[i] <= 0.07)
            {
                benchmark::DoNotOptimize(res += l_extendedprice[i] *
                                                l_discount[i]); // otherwise, the complete loop is optimized away
            }
        }
    }
}

BENCHMARK(Q6)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6_Baseline)->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_TPC_Q1)->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_TPC_Q1Baseline)->Unit(benchmark::kMillisecond);