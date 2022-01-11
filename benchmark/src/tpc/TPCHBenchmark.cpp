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
auto lineitem = CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin.xz"));
auto customer_orders_lineitem =
    CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/customer_orders_lineitem1g_compressed.bin.xz"));
auto part_supplier_lineitem_partsupp_orders_nation =
    CompressedTable::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/q9_wide_table.bin.xz"));

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
    return customer_orders_lineitem.getDictionary(c_mktsegment).at(segments.at(segmentSelect(gen)));
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
    const auto &dict = part_supplier_lineitem_partsupp_orders_nation.getDictionary(p_name);
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

    auto l_quantity = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(lineitem_cols::L_QUANTITY);
    auto l_extendedprice =
        lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(lineitem_cols::L_DISCOUNT);
    auto l_tax = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_TAX>>(lineitem_cols::L_TAX);
    auto l_returnflag = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_RETURNFLAG>>(
        static_cast<const size_t>(lineitem_cols::L_RETURNFLAG));
    auto l_linestatus = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_LINESTATUS>>(lineitem_cols::L_LINESTATUS);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(lineitem_cols::L_SHIPDATE);

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

    auto l_quantity = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(lineitem_cols::L_QUANTITY);
    auto l_extendedprice =
        lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(lineitem_cols::L_DISCOUNT);
    auto l_tax = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_TAX>>(lineitem_cols::L_TAX);
    auto l_returnflag = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_RETURNFLAG>>(lineitem_cols::L_RETURNFLAG);
    auto l_linestatus = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_LINESTATUS>>(lineitem_cols::L_LINESTATUS);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(lineitem_cols::L_SHIPDATE);
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

static void Q3(benchmark::State &state)
{
    using namespace voila;

    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = magic_enum::enum_count<customer_cols>() + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_ORDERKEY>>(
        lineitem_offset + lineitem_cols::L_ORDERKEY);
    auto l_extendedprice = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(
        lineitem_offset + lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(
        lineitem_offset + lineitem_cols::L_DISCOUNT);
    auto l_shipdate = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(
        lineitem_offset + lineitem_cols::L_SHIPDATE);
    auto c_mktsegment =
        customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_MKTSEGMENT>>(customer_cols::C_MKTSEGMENT);
    auto c_custkey =
        customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(customer_cols::C_CUSTKEY);
    auto o_custkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_CUSTKEY>>(orders_offset +
                                                                                                orders_cols::O_CUSTKEY);
    auto o_orderkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERKEY>>(
        orders_offset + orders_cols::O_ORDERKEY);
    auto o_orderdate = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERDATE>>(
        orders_offset + orders_cols::O_ORDERDATE);
    auto o_shippriority = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_SHIPPRIORITY>>(
        orders_offset + orders_cols::O_SHIPPRIORITY);
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

static void Q3_Baseline(benchmark::State &state)
{
    using namespace voila;

    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = magic_enum::enum_count<customer_cols>() + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_ORDERKEY>>(
        lineitem_offset + lineitem_cols::L_ORDERKEY);
    auto l_extendedprice = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(
        lineitem_offset + lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(
        lineitem_offset + lineitem_cols::L_DISCOUNT);
    auto l_shipdate = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(
        lineitem_offset + lineitem_cols::L_SHIPDATE);
    auto c_mktsegment =
        customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_MKTSEGMENT>>(customer_cols::C_MKTSEGMENT);
    auto c_custkey =
        customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(customer_cols::C_CUSTKEY);
    auto o_custkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_CUSTKEY>>(orders_offset +
                                                                                                orders_cols::O_CUSTKEY);
    auto o_orderkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERKEY>>(
        orders_offset + orders_cols::O_ORDERKEY);
    auto o_orderdate = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERDATE>>(
        orders_offset + orders_cols::O_ORDERDATE);
    auto o_shippriority = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_SHIPPRIORITY>>(
        orders_offset + orders_cols::O_SHIPPRIORITY);

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

static void Q6(benchmark::State &state)
{
    using namespace voila;

    auto l_quantity = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(lineitem_cols::L_QUANTITY);
    auto l_extendedprice =
        lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(lineitem_cols::L_DISCOUNT);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(lineitem_cols::L_SHIPDATE);
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

    auto l_quantity = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(lineitem_cols::L_QUANTITY);
    auto l_extendedprice =
        lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(lineitem_cols::L_DISCOUNT);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(lineitem_cols::L_SHIPDATE);

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

static void Q9(benchmark::State &state)
{
    using namespace voila;

    constexpr auto supplier_offset = magic_enum::enum_count<part_cols>();
    constexpr auto lineitem_offset = supplier_offset + magic_enum::enum_count<supplier_cols>();
    constexpr auto partsupp_offset = lineitem_offset + magic_enum::enum_count<lineitem_cols>();
    constexpr auto orders_offset = partsupp_offset + magic_enum::enum_count<partsupp_cols>();
    constexpr auto nations_offset = orders_offset + magic_enum::enum_count<orders_cols>();
    auto n_name = part_supplier_lineitem_partsupp_orders_nation.getColumn<nation_types_t<nation_cols::N_NAME>>(
        nations_offset + nation_cols::N_NAME);
    auto o_orderdate =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<orders_types_t<orders_cols::O_ORDERDATE>>(
            orders_offset + orders_cols::O_ORDERDATE);
    auto l_extendedprice =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(
            lineitem_offset + lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(
            lineitem_offset + lineitem_cols::L_DISCOUNT);
    auto ps_supplycost =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<partsupp_cols::PS_SUPPLYCOST>>(
            partsupp_offset + partsupp_cols::PS_SUPPLYCOST);
    auto l_quantity =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(
            lineitem_offset + lineitem_cols::L_QUANTITY);
    auto s_suppkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<supplier_types_t<supplier_cols::S_SUPPKEY>>(
            supplier_offset + supplier_cols::S_SUPPKEY);
    auto l_suppkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_SUPPKEY>>(
            lineitem_offset + lineitem_cols::L_SUPPKEY);
    auto ps_suppkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<partsupp_cols::PS_SUPPKEY>>(
            partsupp_offset + partsupp_cols::PS_SUPPKEY);
    auto ps_partkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<partsupp_cols::PS_PARTKEY>>(
            partsupp_offset + partsupp_cols::PS_PARTKEY);
    auto l_partkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_PARTKEY>>(
            lineitem_offset + lineitem_cols::L_PARTKEY);
    auto p_partkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<part_types_t<part_cols::P_PARTKEY>>(
        part_cols::P_PARTKEY);
    auto o_orderkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<orders_types_t<orders_cols::O_ORDERKEY>>(
        orders_offset + orders_cols::O_ORDERKEY);
    auto l_orderkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_ORDERKEY>>(
            lineitem_offset + lineitem_cols::L_ORDERKEY);
    auto s_nationkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<supplier_types_t<supplier_cols::S_NATIONKEY>>(
            supplier_offset + supplier_cols::S_NATIONKEY);
    auto n_nationkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<nation_types_t<nation_cols::N_NATIONKEY>>(
            nations_offset + nation_cols::N_NATIONKEY);
    auto p_name = part_supplier_lineitem_partsupp_orders_nation.getColumn<int32_t>(part_cols::P_NAME);

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

static void Q9_Baseline(benchmark::State &state)
{
    using namespace voila;

    constexpr auto supplier_offset = magic_enum::enum_count<part_cols>();
    constexpr auto lineitem_offset = supplier_offset + magic_enum::enum_count<supplier_cols>();
    constexpr auto partsupp_offset = lineitem_offset + magic_enum::enum_count<lineitem_cols>();
    constexpr auto orders_offset = partsupp_offset + magic_enum::enum_count<partsupp_cols>();
    constexpr auto nations_offset = orders_offset + magic_enum::enum_count<orders_cols>();
    auto n_name = part_supplier_lineitem_partsupp_orders_nation.getColumn<nation_types_t<nation_cols::N_NAME>>(
        nations_offset + nation_cols::N_NAME);
    auto o_orderdate =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<orders_types_t<orders_cols::O_ORDERDATE>>(
            orders_offset + orders_cols::O_ORDERDATE);
    auto l_extendedprice =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(
            lineitem_offset + lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(
            lineitem_offset + lineitem_cols::L_DISCOUNT);
    auto ps_supplycost =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<partsupp_cols::PS_SUPPLYCOST>>(
            partsupp_offset + partsupp_cols::PS_SUPPLYCOST);
    auto l_quantity =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(
            lineitem_offset + lineitem_cols::L_QUANTITY);
    auto s_suppkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<supplier_types_t<supplier_cols::S_SUPPKEY>>(
            supplier_offset + supplier_cols::S_SUPPKEY);
    auto l_suppkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_SUPPKEY>>(
            lineitem_offset + lineitem_cols::L_SUPPKEY);
    auto ps_suppkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<partsupp_cols::PS_SUPPKEY>>(
            partsupp_offset + partsupp_cols::PS_SUPPKEY);
    auto ps_partkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<partsupp_cols::PS_PARTKEY>>(
            partsupp_offset + partsupp_cols::PS_PARTKEY);
    auto l_partkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_PARTKEY>>(
            lineitem_offset + lineitem_cols::L_PARTKEY);
    auto p_partkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<part_types_t<part_cols::P_PARTKEY>>(
        part_cols::P_PARTKEY);
    auto o_orderkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<orders_types_t<orders_cols::O_ORDERKEY>>(
        orders_offset + orders_cols::O_ORDERKEY);
    auto l_orderkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<lineitem_cols::L_ORDERKEY>>(
            lineitem_offset + lineitem_cols::L_ORDERKEY);
    auto s_nationkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<supplier_types_t<supplier_cols::S_NATIONKEY>>(
            supplier_offset + supplier_cols::S_NATIONKEY);
    auto n_nationkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<nation_types_t<nation_cols::N_NATIONKEY>>(
            nations_offset + nation_cols::N_NATIONKEY);
    auto p_name = part_supplier_lineitem_partsupp_orders_nation.getColumn<int32_t>(part_cols::P_NAME);

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

static void Q18(benchmark::State &state)
{
    using namespace voila;

    auto customer_orders_lineitem = CompressedTable::readTable(
        std::string(VOILA_BENCHMARK_DATA_PATH "/customer_orders_lineitem1g_compressed.bin.xz"));
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = magic_enum::enum_count<customer_cols>() + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_ORDERKEY>>(
        lineitem_offset + lineitem_cols::L_ORDERKEY);
    auto c_name = customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_NAME>>(customer_cols::C_NAME);
    auto c_custkey =
        customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(customer_cols::C_CUSTKEY);
    auto o_custkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_CUSTKEY>>(orders_offset +
                                                                                                orders_cols::O_CUSTKEY);
    auto o_orderkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERKEY>>(
        orders_offset + orders_cols::O_ORDERKEY);
    auto o_orderdate = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERDATE>>(
        orders_offset + orders_cols::O_ORDERDATE);
    auto o_totalprice = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_TOTALPRICE>>(
        orders_offset + orders_cols::O_TOTALPRICE);
    auto l_quantity = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(
        lineitem_offset + lineitem_cols::L_QUANTITY);
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

static void Q18_Baseline(benchmark::State &state)
{
    using namespace voila;
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = magic_enum::enum_count<customer_cols>() + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_ORDERKEY>>(
        lineitem_offset + lineitem_cols::L_ORDERKEY);
    auto c_name = customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_NAME>>(customer_cols::C_NAME);
    auto c_custkey =
        customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(customer_cols::C_CUSTKEY);
    auto o_custkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_CUSTKEY>>(orders_offset +
                                                                                                orders_cols::O_CUSTKEY);
    auto o_orderkey = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERKEY>>(
        orders_offset + orders_cols::O_ORDERKEY);
    auto o_orderdate = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_ORDERDATE>>(
        orders_offset + orders_cols::O_ORDERDATE);
    auto o_totalprice = customer_orders_lineitem.getColumn<orders_types_t<orders_cols::O_TOTALPRICE>>(
        orders_offset + orders_cols::O_TOTALPRICE);
    auto l_quantity = customer_orders_lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(
        lineitem_offset + lineitem_cols::L_QUANTITY);

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
BENCHMARK(Q3)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6)->Unit(benchmark::kMillisecond);
BENCHMARK(Q9)->Unit(benchmark::kMillisecond);
BENCHMARK(Q18)->Unit(benchmark::kMillisecond);
BENCHMARK(Q1_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q3_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q9_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q18_Baseline)->Unit(benchmark::kMillisecond);