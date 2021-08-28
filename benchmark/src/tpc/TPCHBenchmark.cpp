#include "Config.hpp"
#include "Program.hpp"
#include "Tables.hpp"
#include "benchmark_defs.hpp.inc"

#include <xxhash.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop

std::random_device rd;
std::mt19937 gen(rd());

static int32_t getQ6Date()
{
    constexpr auto dates = std::to_array({19930101, 19930101, 19950101, 19960101, 19970101});
    std::uniform_int_distribution<unsigned int> dateSelect(0, dates.size() - 1);
    return dates[dateSelect(gen)];
}

static double getQ6Discount()
{
    constexpr auto discounts = std::to_array({0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09});
    std::uniform_int_distribution<unsigned int> discountSelect(0, discounts.size() - 1);
    return discounts[discountSelect(gen)];
}

static int32_t getQ1Date()
{
    constexpr auto dates = std::to_array(
        {19981002, 19981001, 19980930, 19980929, 19980928, 19980927, 19980926, 19980925, 19980924, 19980923, 19980922,
         19980921, 19980920, 19980919, 19980918, 19980917, 19980916, 19980915, 19980914, 19980913, 19980912, 19980911,
         19980910, 19980909, 19980908, 19980907, 19980906, 19980905, 19980904, 19980903, 19980902, 19980901, 19980831,
         19980830, 19980829, 19980828, 19980827, 19980826, 19980825, 19980824, 19980823, 19980822, 19980821, 19980820,
         19980819, 19980818, 19980817, 19980816, 19980815, 19980814, 19980813, 19980812, 19980811, 19980810, 19980809,
         19980808, 19980807, 19980806, 19980805, 19980804, 19980803});
    std::uniform_int_distribution<unsigned int> dateSelect(0, dates.size() - 1);
    return dates[dateSelect(gen)];
}

static int32_t getQ6Quantity()
{
    constexpr auto quantities = std::to_array({24, 25});
    std::uniform_int_distribution<unsigned int> quantitySelect(0, quantities.size() - 1);
    return quantities[quantitySelect(gen)];
}

// TODO: global vars and so on...
auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));

static void Q1(benchmark::State &state)
{
    using namespace voila;

    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_tax = lineitem.getColumn<double>(7);
    auto l_returnflag = lineitem.getColumn<int32_t>(8);
    auto l_linestatus = lineitem.getColumn<int32_t>(9);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);

    Config config;
    config.debug = false;
    config.optimize = true;
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q1.voila";
    double queryTime = 0;
    Program prog(query, config);

    for ([[maybe_unused]] auto _ : state)
    {
        auto date = getQ1Date();
        prog << ::voila::make_param(l_returnflag.data(), l_returnflag.size(), DataType::INT32);
        prog << ::voila::make_param(l_linestatus.data(), l_linestatus.size(), DataType::INT32);
        prog << ::voila::make_param(l_quantity.data(), l_quantity.size(), DataType::INT32);
        prog << ::voila::make_param(l_extendedprice.data(), l_extendedprice.size(), DataType::DBL);
        prog << ::voila::make_param(l_discount.data(), l_discount.size(), DataType::DBL);
        prog << ::voila::make_param(l_tax.data(), l_tax.size(), DataType::DBL);
        prog << ::voila::make_param(l_shipdate.data(), l_shipdate.size(), DataType::INT32);
        prog << ::voila::make_param(&date, 0, DataType::INT32);

        // run in jit
        auto res = prog();

        queryTime += prog.getExecTime();
    }
    state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
}

constexpr int32_t INVALID = static_cast<int32_t>(std::numeric_limits<uint64_t>::max());
template<class T1, class T2>
static size_t probeAndInsert(size_t key,
                             const size_t size,
                             const T1 val1,
                             const T2 val2,
                             std::vector<T1> &vals1,
                             std::vector<T2> &vals2)
{
    key %= size;
    // probing
    while (vals1[key % size] != INVALID && !(vals1[key % size] == val1 && vals2[key % size] == val2))
    {
        key += 1;
        key %= size;
    }

    vals1[key] = val1;
    vals2[key] = val2;

    return key;
}

template<class T1, class T2>
static size_t hash(T1 val1, T2 val2)
{
    std::array<char, sizeof(T1) + sizeof(T2)> data{};
    std::copy_n(reinterpret_cast<char *>(&val1), sizeof(T1), data.data());
    std::copy_n(reinterpret_cast<char *>(&val2), sizeof(T2), data.data() + sizeof(T1));
    return XXH3_64bits(data.data(), sizeof(T1) + sizeof(T2));
}

static void Q1_Baseline(benchmark::State &state)
{
    using namespace voila;

    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_tax = lineitem.getColumn<double>(7);
    auto l_returnflag = lineitem.getColumn<int32_t>(8);
    auto l_linestatus = lineitem.getColumn<int32_t>(9);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);
    const auto htSizes = std::bit_ceil(l_quantity.size());
    for ([[maybe_unused]] auto _ : state)
    {
        const auto date = getQ1Date();
        std::vector<int32_t> ht_returnflag(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_linestatus(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int64_t> sum_qty(htSizes, 0);
        std::vector<double> sum_base_price(htSizes, 0);
        std::vector<double> sum_disc_price(htSizes, 0);
        std::vector<double> sum_charge(htSizes, 0);
        std::vector<double> sum_discount(htSizes, 0);
        std::vector<double> avg_qty(htSizes, 0);
        std::vector<double> avg_price(htSizes, 0);
        std::vector<double> avg_disc(htSizes, 0);
        std::vector<int64_t> count_order(htSizes, 0);

        for (size_t i = 0; i < l_quantity.size(); ++i)
        {
            if (l_shipdate[i] <= date)
            {
                const auto idx = probeAndInsert(hash(l_returnflag[i], l_linestatus[i]), htSizes, l_returnflag[i],
                                                l_linestatus[i], ht_returnflag, ht_linestatus);
                sum_qty[idx] += l_quantity[i];
                sum_base_price[idx] += l_extendedprice[i];
                sum_disc_price[idx] += l_extendedprice[i] * (1 - l_discount[i]);
                sum_charge[idx] += l_extendedprice[i] * (1 - l_discount[i]) * (1 + l_tax[i]);
                ++count_order[idx];
                avg_qty[idx] = sum_qty[idx] / count_order[idx];
                avg_price[idx] = sum_base_price[idx] / count_order[idx];
                sum_discount[idx] += l_discount[i];
                avg_disc[idx] = sum_discount[idx] / count_order[idx];
            }
        }
    }
}

static void Q6(benchmark::State &state)
{
    using namespace voila;

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
        auto startDate = getQ6Date();
        auto endDate = startDate + 10000;
        auto quantity = getQ6Quantity();
        auto discount = getQ6Discount();
        auto minDiscount = discount - 0.01;
        auto maxDiscount = discount + 0.01;

        prog << ::voila::make_param(l_quantity.data(), l_quantity.size(), DataType::INT32);
        prog << ::voila::make_param(l_discount.data(), l_discount.size(), DataType::DBL);
        prog << ::voila::make_param(l_shipdate.data(), l_shipdate.size(), DataType::INT32);
        prog << ::voila::make_param(l_extendedprice.data(), l_extendedprice.size(), DataType::DBL);
        prog << ::voila::make_param(&startDate, 0, DataType::INT32);
        prog << ::voila::make_param(&endDate, 0, DataType::INT32);
        prog << ::voila::make_param(&quantity, 0, DataType::INT32);
        prog << ::voila::make_param(&minDiscount, 0, DataType::DBL);
        prog << ::voila::make_param(&maxDiscount, 0, DataType::DBL);

        // run in jit
        auto res = prog();
        queryTime += prog.getExecTime();
    }
    state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
}

static void Q6_Baseline(benchmark::State &state)
{
    using namespace voila;

    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);

    for ([[maybe_unused]] auto _ : state)
    {
        [[maybe_unused]] double res = 0;
        for (size_t i = 0; i < l_quantity.size(); ++i)
        {
            auto startDate = getQ6Date();
            auto endDate = startDate + 10000;
            auto quantity = getQ6Quantity();
            auto discount = getQ6Discount();
            auto minDiscount = discount - 0.01;
            auto maxDiscount = discount + 0.01;
            if (l_shipdate[i] >= startDate && l_shipdate[i] < endDate && l_quantity[i] < quantity &&
                l_discount[i] >= minDiscount && l_discount[i] <= maxDiscount)
            {
                res += l_extendedprice[i] * l_discount[i];
            }
        }
    }
}

BENCHMARK(Q1)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6)->Unit(benchmark::kMillisecond);
BENCHMARK(Q1_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6_Baseline)->Unit(benchmark::kMillisecond);
