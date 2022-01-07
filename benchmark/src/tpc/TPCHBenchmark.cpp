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

// TODO: global vars and so on...
auto part = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/part1g_compressed.bin"));
auto supplier = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/supplier1g_compressed.bin"));
auto partsupp = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/partsupp1g_compressed.bin"));
auto customer = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/customer1g_compressed.bin"));
auto orders = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/orders1g_compressed.bin"));
auto lineitem = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
auto nation = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/nation1g_compressed.bin"));
auto region = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/region1g_compressed.bin"));
auto wide_customer_orders_lineitem = CompressedTable::makeWideTable(
    {customer, orders, lineitem},
    {std::make_pair(static_cast<size_t>(customer_cols::C_CUSTKEY), static_cast<size_t>(orders_cols::O_CUSTKEY)),
     std::make_pair(static_cast<size_t>(orders_cols::O_ORDERKEY), static_cast<size_t>(lineitem_cols::L_ORDERKEY))});

// substitution parameter generators
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

[[maybe_unused]] static int32_t getQ3Segment()
{
    constexpr auto segments = std::to_array({"AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"});
    constexpr size_t c_mktsegment = 6;
    std::uniform_int_distribution<unsigned int> segmentSelect(0, segments.size() - 1);
    return customer.getDictionary(c_mktsegment).at(segments.at(segmentSelect(gen)));
}

[[maybe_unused]] static int32_t getQ3Date()
{
    constexpr auto dates = std::to_array(
        {19950301, 19950302, 19950303, 19950304, 19950305, 19950306, 19950307, 19950308, 19950309, 19950310, 19950311,
         19950312, 19950313, 19950314, 19950315, 19950316, 19950317, 19950318, 19950319, 19950320, 19950321, 19950322,
         19950323, 19950324, 19950325, 19950326, 19950327, 19950328, 19950329, 19950330, 19950331});
    std::uniform_int_distribution<unsigned int> dateSelect(0, dates.size() - 1);
    return dates[dateSelect(gen)];
}

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

static int32_t getQ6Quantity()
{
    constexpr auto quantities = std::to_array({24, 25});
    std::uniform_int_distribution<unsigned int> quantitySelect(0, quantities.size() - 1);
    return quantities[quantitySelect(gen)];
}

[[maybe_unused]] static std::vector<int32_t> getQ9Color()
{
    constexpr auto colors = std::to_array(
        {"almond",    "antique",    "aquamarine", "azure",     "beige",     "bisque",     "black",     "blanched",
         "blue",      "blush",      "brown",      "burlywood", "burnished", "chartreuse", "chiffon",   "chocolate",
         "coral",     "cornflower", "cornsilk",   "cream",     "cyan",      "dark",       "deep",      "dim",
         "dodger",    "drab",       "firebrick",  "floral",    "forest",    "frosted",    "gainsboro", "ghost",
         "goldenrod", "green",      "grey",       "honeydew",  "hot",       "indian",     "ivory",     "khaki",
         "lace",      "lavender",   "lawn",       "lemon",     "light",     "lime",       "linen",     "magenta",
         "maroon",    "medium",     "metallic",   "midnight",  "mint",      "misty",      "moccasin",  "navajo",
         "navy",      "olive",      "orange",     "orchid",    "pale",      "papaya",     "peach",     "peru",
         "pink",      "plum",       "powder",     "puff",      "purple",    "red",        "rose",      "rosy",
         "royal",     "saddle",     "salmon",     "sandy",     "seashell",  "sienna",     "sky",       "slate",
         "smoke",     "snow",       "spring",     "steel",     "tan",       "thistle",    "tomato",    "turquoise",
         "violet",    "wheat",      "white",      "yellow"});
    std::uniform_int_distribution<unsigned int> colorSelect(0, colors.size() - 1);
    const auto color = colors.at(colorSelect(gen));
    constexpr size_t p_name = 1;
    const auto &dict = part.getDictionary(p_name);
    std::vector<int32_t> colorSet;
    for (auto &elem : dict)
    {
        if (elem.first.find(color) != std::string::npos)
        {
            colorSet.push_back(elem.second);
        }
    }

    return colorSet;
}

[[maybe_unused]] static int32_t getQ18Quantity()
{
    std::uniform_int_distribution<int32_t> quantitySelect(312, 315);
    return quantitySelect(gen);
}

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
    config.tile = false;
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q1.voila";
    double queryTime = 0;
    Program prog(query, config);

    for ([[maybe_unused]] auto _ : state)
    {
        auto date = getQ1Date();
        prog << l_returnflag;
        prog << l_linestatus;
        prog << l_quantity;
        prog << l_extendedprice;
        prog << l_discount;
        prog << l_tax;
        prog << l_shipdate;
        prog << &date;

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

        prog << l_quantity;
        prog << l_discount;
        prog << l_shipdate;
        prog << l_extendedprice;
        prog << &startDate;
        prog << &endDate;
        prog << &quantity;
        prog << &minDiscount;
        prog << &maxDiscount;

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
        const auto startDate = getQ6Date();
        const auto endDate = startDate + 10000;
        const auto quantity = getQ6Quantity();
        const auto discount = getQ6Discount();
        const auto minDiscount = discount - 0.01;
        const auto maxDiscount = discount + 0.01;
        double res = 0;
        for (size_t i = 0; i < l_quantity.size(); ++i)
        {
            if (l_shipdate[i] >= startDate && l_shipdate[i] < endDate && l_quantity[i] < quantity &&
                l_discount[i] >= minDiscount && l_discount[i] <= maxDiscount)
            {
                ::benchmark::DoNotOptimize(res += l_extendedprice[i] * l_discount[i]);
            }
        }
    }
}

BENCHMARK(Q1)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6)->Unit(benchmark::kMillisecond);
BENCHMARK(Q1_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6_Baseline)->Unit(benchmark::kMillisecond);
