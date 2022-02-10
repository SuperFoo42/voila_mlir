#include "BenchmarkState.hpp"
#include "Config.hpp"
#include "Program.hpp"
#include "Tables.hpp"
#include "benchmark_defs.hpp.inc"
#include "no_partitioning_join.hpp"

#include <xxhash.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"

#include <benchmark/benchmark.h>

#pragma GCC diagnostic pop

std::random_device rd;
std::mt19937 gen(rd());
enum ArgumentTypes
{
    THREAD_COUNT,
    TILING,
    PEELING,
    VECTORIZE,
    VECTOR_SIZE,
    UNROLL_FACTOR,
    PARALLELIZE_TYPE
};

extern std::unique_ptr<BenchmarkState> benchmarkState;

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
    return benchmarkState->getCustomerOrderLineitem().getDictionary(c_mktsegment).at(segments.at(segmentSelect(gen)));
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
    const auto &dict = benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getDictionary(p_name);
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

    auto l_quantity = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto l_extendedprice =
        benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_tax = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_TAX>>(L_TAX);
    auto l_returnflag = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_RETURNFLAG>>(
        static_cast<const size_t>(L_RETURNFLAG));
    auto l_linestatus = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_LINESTATUS>>(L_LINESTATUS);
    auto l_shipdate = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_SHIPDATE>>(L_SHIPDATE);

    Config config;
    config.optimize()
        .debug(true)
        .tile(state.range(TILING))
        .peel(state.range(PEELING))
        .vectorize(state.range(VECTORIZE))
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) != 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .unroll_factor(state.range(UNROLL_FACTOR))
        .parallel_threads(state.range(THREAD_COUNT));
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q1.voila";
    double queryTime = 0;
    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto date = getQ1Date();
        state.ResumeTiming();
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
        // cleanup
        state.PauseTiming();
        /* for (auto &el : res)
         {
             switch (el.index())
             {
                 case 0 */
        /*strided_memref_ptr<uint32_t, 1>*/ /*:
    std::free(std::get<strided_memref_ptr<uint32_t, 1>>(el).get()->basePtr);
    break;
case 1 */
        /*strided_memref_ptr<uint64_t, 1>*/ /*:
    std::free(std::get<strided_memref_ptr<uint64_t, 1>>(el).get()->basePtr);
    break;
case 2 */
        /*strided_memref_ptr<double, 1>*/   /*:
      std::free(std::get<strided_memref_ptr<double, 1>>(el).get()->basePtr);
      break;
  default:
      continue;
}
}*/
        state.ResumeTiming();
        queryTime += prog.getExecTime();
    }
    state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
}

constexpr int32_t INVALID = static_cast<int32_t>(std::numeric_limits<uint64_t>::max());

template<class T>
static size_t hash(T val1)
{
    return XXH3_64bits(&val1, sizeof(T));
}
template<class T1, class T2>
static size_t hash(T1 val1, T2 val2)
{
    std::array<char, sizeof(T1) + sizeof(T2)> data{};
    std::copy_n(reinterpret_cast<char *>(&val1), sizeof(T1), data.data());
    std::copy_n(reinterpret_cast<char *>(&val2), sizeof(T2), data.data() + sizeof(T1));
    return XXH3_64bits(data.data(), data.size());
}

template<class T1, class T2, class T3>
static size_t hash(T1 val1, T2 val2, T3 val3)
{
    std::array<char, sizeof(T1) + sizeof(T2) + sizeof(T3)> data{};
    std::copy_n(reinterpret_cast<char *>(&val1), sizeof(T1), data.data());
    std::copy_n(reinterpret_cast<char *>(&val2), sizeof(T2), data.data() + sizeof(T1));
    std::copy_n(reinterpret_cast<char *>(&val3), sizeof(T3), data.data() + sizeof(T1) + sizeof(T2));
    return XXH3_64bits(data.data(), data.size());
}

template<class T1, class T2, class T3, class T4, class T5>
static size_t hash(T1 val1, T2 val2, T3 val3, T4 val4, T5 val5)
{
    std::array<char, sizeof(T1) + sizeof(T2) + sizeof(T3) + sizeof(T4) + sizeof(T5)> data{};
    std::copy_n(reinterpret_cast<char *>(&val1), sizeof(T1), data.data());
    std::copy_n(reinterpret_cast<char *>(&val2), sizeof(T2), data.data() + sizeof(T1));
    std::copy_n(reinterpret_cast<char *>(&val3), sizeof(T3), data.data() + sizeof(T1) + sizeof(T2));
    std::copy_n(reinterpret_cast<char *>(&val4), sizeof(T4), data.data() + sizeof(T1) + sizeof(T2) + sizeof(T3));
    std::copy_n(reinterpret_cast<char *>(&val5), sizeof(T5),
                data.data() + sizeof(T1) + sizeof(T2) + sizeof(T3) + sizeof(T4));
    return XXH3_64bits(data.data(), data.size());
}

template<class T1, class T2>
static size_t probeAndInsert(size_t key, const T1 val1, const T2 val2, std::vector<T1> &vals1, std::vector<T2> &vals2)
{
    assert(vals1.size() == vals2.size());
    const auto size = vals1.size();
    key %= size;
    // probing
    while (vals1[key] != INVALID && !(vals1[key] == val1 && vals2[key] == val2))
    {
        key += 1;
        key %= size;
    }

    vals1[key] = val1;
    vals2[key] = val2;

    return key;
}

template<class T>
static size_t contains(size_t key, T val, std::vector<T> &vals)
{
    const auto size = vals.size();
    key %= size;
    // probing
    while (vals[key % size] != INVALID && vals[key % size] != val)
    {
        key += 1;
        key %= size;
    }

    return vals[key % size] == val;
}

template<class T1, class T2, class T3>
static size_t probeAndInsert(size_t key,
                             const T1 val1,
                             const T2 val2,
                             const T3 val3,
                             std::vector<T1> &vals1,
                             std::vector<T2> &vals2,
                             std::vector<T3> &vals3)
{
    assert(vals1.size() == vals2.size() && vals2.size() == vals3.size());
    const auto size = vals1.size();
    key %= size;
    // probing
    while (vals1[key] != INVALID && !(vals1[key] == val1 && vals2[key] == val2 && vals3[key] == val3))
    {
        key += 1;
        key %= size;
    }

    vals1[key] = val1;
    vals2[key] = val2;
    vals3[key] = val3;

    return key;
}

template<class T1>
static size_t probeAndInsert(size_t key, const T1 val1, std::vector<T1> &vals1)
{
    key &= vals1.size() - 1;
    // probing
    while (vals1[key] != INVALID && !(vals1[key] == val1))
    {
        key += 1;
        key &= vals1.size() - 1;
    }

    vals1[key] = val1;

    return key;
}

template<class T1, class T2, class T3, class T4, class T5>
static size_t probeAndInsert(size_t key,
                             const T1 val1,
                             const T2 val2,
                             const T3 val3,
                             const T4 val4,
                             const T5 val5,
                             std::vector<T1> &vals1,
                             std::vector<T2> &vals2,
                             std::vector<T3> &vals3,
                             std::vector<T4> &vals4,
                             std::vector<T5> &vals5)
{
    assert(vals1.size() == vals2.size() && vals2.size() == vals3.size() && vals3.size() == vals4.size() &&
           vals4.size() == vals5.size());
    const auto size = vals1.size();
    key %= size;
    // probing
    while (vals1[key] != INVALID && !(vals1[key] == val1 && vals2[key] == val2 && vals3[key] == val3 &&
                                      vals4[key] == val4 && vals5[key] == val5))
    {
        key += 1;
        key %= size;
    }

    vals1[key] = val1;
    vals2[key] = val2;
    vals3[key] = val3;
    vals4[key] = val4;
    vals5[key] = val5;

    return key;
}

static void Q1_Baseline(benchmark::State &state)
{
    using namespace voila;

    auto l_quantity = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto l_extendedprice =
        benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_tax = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_TAX>>(L_TAX);
    auto l_returnflag = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_RETURNFLAG>>(L_RETURNFLAG);
    auto l_linestatus = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_LINESTATUS>>(L_LINESTATUS);
    auto l_shipdate = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_SHIPDATE>>(L_SHIPDATE);
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
                const auto idx = probeAndInsert(hash(l_returnflag[i], l_linestatus[i]), l_returnflag[i],
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
    auto l_orderkey = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_ORDERKEY>>(
        lineitem_offset + L_ORDERKEY);
    auto l_extendedprice = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(
        lineitem_offset + L_EXTENDEDPRICE);
    auto l_discount = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_DISCOUNT>>(
        lineitem_offset + L_DISCOUNT);
    auto l_shipdate = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_SHIPDATE>>(
        lineitem_offset + L_SHIPDATE);
    auto c_mktsegment =
        benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<C_MKTSEGMENT>>(C_MKTSEGMENT);
    auto c_custkey = benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_CUSTKEY>>(orders_offset + O_CUSTKEY);
    auto o_orderkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto o_orderdate =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERDATE>>(orders_offset + O_ORDERDATE);
    auto o_shippriority = benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_SHIPPRIORITY>>(
        orders_offset + O_SHIPPRIORITY);

    Config config;
    config.optimize()
        .debug(false)
        .tile(state.range(TILING))
        .peel(state.range(PEELING))
        .vectorize(state.range(VECTORIZE))
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) != 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .unroll_factor(state.range(UNROLL_FACTOR))
        .parallel_threads(state.range(THREAD_COUNT));
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q3.voila";
    Program prog(query, config);
    double queryTime = 0;
    for ([[maybe_unused]] auto _ : state)
    {
        // qualification data
        state.PauseTiming();
        int32_t segment = getQ3Segment();
        int32_t date = getQ3Date();
        state.ResumeTiming();
        // voila calculations

        prog << l_orderkey;
        prog << l_extendedprice;
        prog << l_discount;
        prog << l_shipdate;
        prog << c_mktsegment;
        prog << c_custkey;
        prog << o_custkey;
        prog << o_orderkey;
        prog << o_orderdate;
        prog << o_shippriority;
        prog << &segment;
        prog << &date;

        auto res = prog();
        state.PauseTiming();
        for (auto &el : res)
        {
            switch (el.index())
            {
                case 0 /*strided_memref_ptr<uint32_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint32_t, 1>>(el).get()->basePtr);
                    break;
                case 1 /*strided_memref_ptr<uint64_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint64_t, 1>>(el).get()->basePtr);
                    break;
                case 2 /*strided_memref_ptr<double, 1>*/:
                    std::free(std::get<strided_memref_ptr<double, 1>>(el).get()->basePtr);
                    break;
                default:
                    continue;
            }
        }
        state.ResumeTiming();
        queryTime += prog.getExecTime();
    }
    state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
}

static void Q3_Baseline(benchmark::State &state)
{
    using namespace voila;
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = magic_enum::enum_count<customer_cols>() + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_ORDERKEY>>(
        lineitem_offset + L_ORDERKEY);
    auto l_extendedprice = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(
        lineitem_offset + L_EXTENDEDPRICE);
    auto l_discount = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_DISCOUNT>>(
        lineitem_offset + L_DISCOUNT);
    auto l_shipdate = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_SHIPDATE>>(
        lineitem_offset + L_SHIPDATE);
    auto c_mktsegment =
        benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<C_MKTSEGMENT>>(C_MKTSEGMENT);
    auto c_custkey = benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_CUSTKEY>>(orders_offset + O_CUSTKEY);
    auto o_orderkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto o_orderdate =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERDATE>>(orders_offset + O_ORDERDATE);
    auto o_shippriority = benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_SHIPPRIORITY>>(
        orders_offset + O_SHIPPRIORITY);

    for ([[maybe_unused]] auto _ : state)
    {
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_o_shippriority_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<double> sum_disc_price_ref(htSizes, 0);
        int32_t segment = getQ3Segment();
        int32_t date = getQ3Date();
        for (size_t i = 0; i < l_orderkey.size(); ++i)
        {
            if (c_mktsegment[i] == segment && c_custkey[i] == o_custkey[i] && l_orderkey[i] == o_orderkey[i] &&
                o_orderdate[i] < date && l_shipdate[i] > date)
            {
                const auto idx = probeAndInsert(hash(l_orderkey[i], o_orderdate[i], o_shippriority[i]), l_orderkey[i],
                                                o_orderdate[i], o_shippriority[i], ht_l_orderkey_ref,
                                                ht_o_orderdate_ref, ht_o_shippriority_ref);

                ::benchmark::DoNotOptimize(sum_disc_price_ref[idx] += l_extendedprice[i] * (1 - l_discount[i]));
            }
        }
    }
}

static void Q3_JoinCompressed(benchmark::State &state)
{
    using namespace voila;
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = magic_enum::enum_count<customer_cols>() + magic_enum::enum_count<orders_cols>();
    auto l_orderkey =
        benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_ORDERKEY>>(lineitem_offset + L_ORDERKEY);
    auto l_extendedprice = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(
        lineitem_offset + L_EXTENDEDPRICE);
    auto l_discount =
        benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_DISCOUNT>>(lineitem_offset + L_DISCOUNT);
    auto l_shipdate =
        benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_SHIPDATE>>(lineitem_offset + L_SHIPDATE);
    auto c_mktsegment = benchmarkState->getCustomerCompressed().getColumn<customer_types_t<C_MKTSEGMENT>>(C_MKTSEGMENT);
    auto c_custkey = benchmarkState->getCustomerCompressed().getColumn<customer_types_t<C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey =
        benchmarkState->getOrdersCompressed().getColumn<orders_types_t<O_CUSTKEY>>(orders_offset + O_CUSTKEY);
    auto o_orderkey =
        benchmarkState->getOrdersCompressed().getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto o_orderdate =
        benchmarkState->getOrdersCompressed().getColumn<orders_types_t<O_ORDERDATE>>(orders_offset + O_ORDERDATE);
    auto o_shippriority =
        benchmarkState->getOrdersCompressed().getColumn<orders_types_t<O_SHIPPRIORITY>>(orders_offset + O_SHIPPRIORITY);

    for ([[maybe_unused]] auto _ : state)
    {
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_o_shippriority_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<double> sum_disc_price_ref(htSizes, 0);
        int32_t segment = getQ3Segment();
        int32_t date = getQ3Date();
        const auto jt = joins::NPO_st(
            l_orderkey, [&l_shipdate, date](auto idx, auto) { return l_shipdate[idx] > date; }, o_orderkey,
            [&o_orderdate, date](auto idx, auto) { return o_orderdate[idx] < date; });
        decltype(o_custkey) m_custkey;
        m_custkey.reserve(jt.size());
        for (auto &el : jt)
        {
            m_custkey.push_back(o_custkey[el.second]);
        }
        const auto jt2 = joins::NPO_st(
            m_custkey, [](auto, auto) { return true; }, c_custkey,
            [&c_mktsegment, segment](auto idx, auto) { return c_mktsegment[idx] == segment; });

        for (const auto &e : jt2)
        {
            const auto l_idx = jt[e.first].first;
            const auto o_idx = jt[e.first].second;
            const auto idx = probeAndInsert(hash(l_orderkey[l_idx], o_orderdate[o_idx], o_shippriority[o_idx]),
                                            l_orderkey[l_idx], o_orderdate[o_idx], o_shippriority[o_idx],
                                            ht_l_orderkey_ref, ht_o_orderdate_ref, ht_o_shippriority_ref);

            benchmark::DoNotOptimize(sum_disc_price_ref[idx] += l_extendedprice[l_idx] * (1 - l_discount[l_idx]));
        }
    }
}

static void Q6(benchmark::State &state)
{
    using namespace voila;

    auto l_quantity = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto l_extendedprice =
        benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_shipdate = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_SHIPDATE>>(L_SHIPDATE);
    Config config;
    config.optimize()
        .debug(false)
        .tile(state.range(TILING))
        .peel(state.range(PEELING))
        .vectorize(state.range(VECTORIZE))
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) != 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .parallel_threads(state.range(THREAD_COUNT))
        .unroll_factor(state.range(UNROLL_FACTOR));

    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q6.voila";

    double queryTime = 0;

    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto startDate = getQ6Date();
        auto endDate = startDate + 10000;
        auto quantity = getQ6Quantity();
        auto discount = getQ6Discount();
        auto minDiscount = discount - 0.01;
        auto maxDiscount = discount + 0.01;
        state.ResumeTiming();
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
        state.PauseTiming();
        for (auto &el : res)
        {
            switch (el.index())
            {
                case 0 /*strided_memref_ptr<uint32_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint32_t, 1>>(el).get()->basePtr);
                    break;
                case 1 /*strided_memref_ptr<uint64_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint64_t, 1>>(el).get()->basePtr);
                    break;
                case 2 /*strided_memref_ptr<double, 1>*/:
                    std::free(std::get<strided_memref_ptr<double, 1>>(el).get()->basePtr);
                    break;
                default:
                    continue;
            }
        }
        state.ResumeTiming();
        queryTime += prog.getExecTime();
    }
    state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
}

static void Q6_Baseline(benchmark::State &state)
{
    using namespace voila;

    auto l_quantity = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto l_extendedprice =
        benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_shipdate = benchmarkState->getLineitemCompressed().getColumn<lineitem_types_t<L_SHIPDATE>>(L_SHIPDATE);

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
    auto n_name = benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<nation_types_t<N_NAME>>(
        nations_offset + N_NAME);
    auto o_orderdate =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<orders_types_t<O_ORDERDATE>>(
            orders_offset + O_ORDERDATE);
    auto l_extendedprice =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(
            lineitem_offset + L_EXTENDEDPRICE);
    auto l_discount =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_DISCOUNT>>(
            lineitem_offset + L_DISCOUNT);
    auto ps_supplycost =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<partsupp_types_t<PS_SUPPLYCOST>>(
            partsupp_offset + PS_SUPPLYCOST);
    auto l_quantity =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_QUANTITY>>(
            lineitem_offset + L_QUANTITY);
    auto s_suppkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<supplier_types_t<S_SUPPKEY>>(
            supplier_offset + S_SUPPKEY);
    auto l_suppkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_SUPPKEY>>(
            lineitem_offset + L_SUPPKEY);
    auto ps_suppkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<partsupp_types_t<PS_SUPPKEY>>(
            partsupp_offset + PS_SUPPKEY);
    auto ps_partkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<partsupp_types_t<PS_PARTKEY>>(
            partsupp_offset + PS_PARTKEY);
    auto l_partkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_PARTKEY>>(
            lineitem_offset + L_PARTKEY);
    auto p_partkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<part_types_t<P_PARTKEY>>(P_PARTKEY);
    auto o_orderkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<orders_types_t<O_ORDERKEY>>(
            orders_offset + O_ORDERKEY);
    auto l_orderkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_ORDERKEY>>(
            lineitem_offset + L_ORDERKEY);
    auto s_nationkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<supplier_types_t<S_NATIONKEY>>(
            supplier_offset + S_NATIONKEY);
    auto n_nationkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<nation_types_t<N_NATIONKEY>>(
            nations_offset + N_NATIONKEY);
    auto p_name = benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<int32_t>(P_NAME);

    Config config;
    config.optimize()
        .debug(false)
        .tile(state.range(TILING))
        .peel(state.range(PEELING))
        .vectorize(state.range(VECTORIZE))
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) != 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .parallel_threads(state.range(THREAD_COUNT))
        .unroll_factor(state.range(UNROLL_FACTOR));

    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q9.voila";

    double queryTime = 0;

    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto needle = getQ9Color();
        state.ResumeTiming();
        prog << n_name;
        prog << o_orderdate;
        prog << l_extendedprice;
        prog << l_discount;
        prog << ps_supplycost;
        prog << l_quantity;
        prog << s_suppkey;
        prog << l_suppkey;
        prog << ps_suppkey;
        prog << ps_partkey;
        prog << l_partkey;
        prog << p_partkey;
        prog << o_orderkey;
        prog << l_orderkey;
        prog << s_nationkey;
        prog << n_nationkey;
        prog << p_name;
        prog << needle;

        auto res = prog();
        state.PauseTiming();
        for (auto &el : res)
        {
            switch (el.index())
            {
                case 0 /*strided_memref_ptr<uint32_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint32_t, 1>>(el).get()->basePtr);
                    break;
                case 1 /*strided_memref_ptr<uint64_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint64_t, 1>>(el).get()->basePtr);
                    break;
                case 2 /*strided_memref_ptr<double, 1>*/:
                    std::free(std::get<strided_memref_ptr<double, 1>>(el).get()->basePtr);
                    break;
                default:
                    continue;
            }
        }
        state.ResumeTiming();
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
    auto n_name = benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<nation_types_t<N_NAME>>(
        nations_offset + N_NAME);
    auto o_orderdate =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<orders_types_t<O_ORDERDATE>>(
            orders_offset + O_ORDERDATE);
    auto l_extendedprice =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(
            lineitem_offset + L_EXTENDEDPRICE);
    auto l_discount =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_DISCOUNT>>(
            lineitem_offset + L_DISCOUNT);
    auto ps_supplycost =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<partsupp_types_t<PS_SUPPLYCOST>>(
            partsupp_offset + PS_SUPPLYCOST);
    auto l_quantity =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_QUANTITY>>(
            lineitem_offset + L_QUANTITY);
    auto s_suppkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<supplier_types_t<S_SUPPKEY>>(
            supplier_offset + S_SUPPKEY);
    auto l_suppkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_SUPPKEY>>(
            lineitem_offset + L_SUPPKEY);
    auto ps_suppkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<partsupp_types_t<PS_SUPPKEY>>(
            partsupp_offset + PS_SUPPKEY);
    auto ps_partkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<partsupp_types_t<PS_PARTKEY>>(
            partsupp_offset + PS_PARTKEY);
    auto l_partkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_PARTKEY>>(
            lineitem_offset + L_PARTKEY);
    auto p_partkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<part_types_t<P_PARTKEY>>(P_PARTKEY);
    auto o_orderkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<orders_types_t<O_ORDERKEY>>(
            orders_offset + O_ORDERKEY);
    auto l_orderkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<lineitem_types_t<L_ORDERKEY>>(
            lineitem_offset + L_ORDERKEY);
    auto s_nationkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<supplier_types_t<S_NATIONKEY>>(
            supplier_offset + S_NATIONKEY);
    auto n_nationkey =
        benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<nation_types_t<N_NATIONKEY>>(
            nations_offset + N_NATIONKEY);
    auto p_name = benchmarkState->getPartSupplierLineitemPartsuppOrdersNation().getColumn<int32_t>(P_NAME);

    // qualification data
    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto needle = getQ9Color();
        state.ResumeTiming();
        // reference impl
        // double ref = 0;
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_n_name_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_needle_ref(std::bit_ceil(needle.size()), static_cast<int32_t>(INVALID));
        std::vector<double> sum_disc_price_ref(htSizes, 0);

        for (const auto &elem : needle)
        {
            auto h = hash(elem);
            probeAndInsert(h, elem, ht_needle_ref);
        }

        for (size_t i = 0; i < l_orderkey.size(); ++i)
        {
            if (s_nationkey[i] == n_nationkey[i] && o_orderkey[i] == l_orderkey[i] && p_partkey[i] == l_partkey[i] &&
                ps_partkey[i] == l_partkey[i] && ps_suppkey[i] == l_suppkey[i] && s_suppkey[i] == l_suppkey[i] &&
                contains(hash(p_name[i]), p_name[i], ht_needle_ref))
            {
                const auto idx = probeAndInsert(hash(n_name[i], o_orderdate[i] / 10000), n_name[i],
                                                o_orderdate[i] / 10000, ht_n_name_ref, ht_o_orderdate_ref);
                benchmark::DoNotOptimize(sum_disc_price_ref[idx] +=
                                         l_extendedprice[i] * (1 - l_discount[i]) - ps_supplycost[i] * l_quantity[i]);
            }
        }
    }
}

static void Q18(benchmark::State &state)
{
    using namespace voila;
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = orders_offset + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_ORDERKEY>>(
        lineitem_offset + L_ORDERKEY);
    auto c_name = benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<C_NAME>>(C_NAME);
    auto c_custkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_CUSTKEY>>(orders_offset + O_CUSTKEY);
    auto o_orderkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto o_orderdate =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERDATE>>(orders_offset + O_ORDERDATE);
    auto o_totalprice = benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_TOTALPRICE>>(
        orders_offset + O_TOTALPRICE);
    auto l_quantity = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_QUANTITY>>(
        lineitem_offset + L_QUANTITY);

    // qualification data

    // voila calculations
    Config config;
    config.optimize()
        .debug(false)
        .tile(state.range(TILING))
        .peel(state.range(PEELING))
        .vectorize(state.range(VECTORIZE))
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) != 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .parallel_threads(state.range(THREAD_COUNT))
        .unroll_factor(state.range(UNROLL_FACTOR));
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q18.voila";
    Program prog(query, config);

    double queryTime = 0;

    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto quantity = getQ18Quantity();
        state.ResumeTiming();
        prog << c_name;
        prog << c_custkey;
        prog << o_orderkey;
        prog << o_orderdate;
        prog << o_totalprice;
        prog << l_quantity;
        prog << l_orderkey;
        prog << o_custkey;
        prog << &quantity;

        auto res = prog();
        state.PauseTiming();
        for (auto &el : res)
        {
            switch (el.index())
            {
                case 0 /*strided_memref_ptr<uint32_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint32_t, 1>>(el).get()->basePtr);
                    break;
                case 1 /*strided_memref_ptr<uint64_t, 1>*/:
                    std::free(std::get<strided_memref_ptr<uint64_t, 1>>(el).get()->basePtr);
                    break;
                case 2 /*strided_memref_ptr<double, 1>*/:
                    std::free(std::get<strided_memref_ptr<double, 1>>(el).get()->basePtr);
                    break;
                default:
                    continue;
            }
        }
        state.ResumeTiming();
        queryTime += prog.getExecTime();
    }
    state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
}

static void Q18_Baseline(benchmark::State &state)
{
    using namespace voila;
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = orders_offset + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_ORDERKEY>>(
        lineitem_offset + L_ORDERKEY);
    auto c_name = benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<C_NAME>>(C_NAME);
    auto c_custkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_CUSTKEY>>(orders_offset + O_CUSTKEY);
    auto o_orderkey =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto o_orderdate =
        benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_ORDERDATE>>(orders_offset + O_ORDERDATE);
    auto o_totalprice = benchmarkState->getCustomerOrderLineitem().getColumn<orders_types_t<O_TOTALPRICE>>(
        orders_offset + O_TOTALPRICE);
    auto l_quantity = benchmarkState->getCustomerOrderLineitem().getColumn<lineitem_types_t<L_QUANTITY>>(
        lineitem_offset + L_QUANTITY);
    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto quantity = getQ18Quantity();
        state.ResumeTiming();
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_c_name_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_c_custkey_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_o_orderkey_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<double> ht_o_totalprice_ref(htSizes, static_cast<int32_t>(INVALID));
        std::vector<double> sum_l_quantity(htSizes, 0);
        std::vector<double> sum_l_quantity2(htSizes, 0);
        std::vector<size_t> orderkey_idxs(l_orderkey.size());

        for (size_t i = 0; i < l_orderkey.size(); ++i)
        {
            const auto idx = probeAndInsert(hash(l_orderkey[i]), l_orderkey[i], ht_l_orderkey_ref);
            sum_l_quantity[idx] += l_quantity[i];
            orderkey_idxs[i] = idx;
        }

        for (size_t i = 0; i < l_orderkey.size(); ++i)
        {
            if (sum_l_quantity[orderkey_idxs[i]] > quantity)
            {
                const auto idx = probeAndInsert(
                    hash(c_name[i], c_custkey[i], o_orderkey[i], o_orderdate[i], o_totalprice[i]), c_name[i],
                    c_custkey[i], o_orderkey[i], o_orderdate[i], o_totalprice[i], ht_c_name_ref, ht_c_custkey_ref,
                    ht_o_orderkey_ref, ht_o_orderdate_ref, ht_o_totalprice_ref);
                sum_l_quantity2[idx] += l_quantity[i];
            }
        }
    }
}

BENCHMARK(Q1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ArgsProduct({/*thread count*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*peeling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorsize*/ benchmark::CreateRange(4, 32, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*off/async/openmp*/ benchmark::CreateDenseRange(0, 2, 1)});
BENCHMARK(Q3)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({/*thread count*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*peeling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorsize*/ benchmark::CreateRange(4, 32, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*off/async/openmp*/ benchmark::CreateDenseRange(0, 2, 1)});
BENCHMARK(Q6)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({/*thread count*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*peeling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(4, 32, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*parallelize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*off/async/openmp*/ benchmark::CreateDenseRange(0, 2, 1)});
BENCHMARK(Q9)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({/*thread count*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*peeling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(4, 32, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*parallelize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*off/async/openmp*/ benchmark::CreateDenseRange(0, 2, 1)});
BENCHMARK(Q18)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({/*thread count*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*peeling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(4, 32, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency() * 2, 2),
                   /*parallelize*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*off/async/openmp*/ benchmark::CreateDenseRange(0, 2, 1)});
BENCHMARK(Q1_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q3_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q3_JoinCompressed)->Unit(benchmark::kMillisecond);
BENCHMARK(Q6_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q9_Baseline)->Unit(benchmark::kMillisecond);
BENCHMARK(Q18_Baseline)->Unit(benchmark::kMillisecond);