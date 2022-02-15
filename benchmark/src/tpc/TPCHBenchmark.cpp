#include "BenchmarkState.hpp"
#include "BenchmarkUtils.hpp"
#include "Config.hpp"
#include "Program.hpp"
#include "QueryGenerator.hpp"
#include "Tables.hpp"
#include "no_partitioning_join.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"

#include <benchmark/benchmark.h>
#include <range/v3/all.hpp>
#pragma GCC diagnostic pop

extern std::unique_ptr<BenchmarkState> benchmarkState;
extern std::unique_ptr<QueryGenerator> queryGenerator;
extern int iterations;
enum ArgumentTypes
{
    TILING,
    VECTOR_SIZE,
    UNROLL_FACTOR,
    PARALLELIZE_TYPE,
    OPTIMIZE_SELECTIONS
};
static void Q1(benchmark::State &state)
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

    Config config;
    config.optimize()
        .debug(false)
        .tile(state.range(TILING) > 0)
        .optimize_selections(state.range(OPTIMIZE_SELECTIONS) > 1)
        .vectorize(state.range(VECTOR_SIZE) > 1)
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) == 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .unroll_factor(state.range(UNROLL_FACTOR));
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q1.voila";
    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        auto date = queryGenerator->getQ1Date();
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

        state.SetIterationTime(prog.getExecTime() / 1000);
    }
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
        const auto date = queryGenerator->getQ1Date();
        std::vector<int32_t> ht_returnflag(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_linestatus(htSizes, INVALID<int32_t>::val);
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
        .tile(state.range(TILING) > 0)
        .optimize_selections(state.range(OPTIMIZE_SELECTIONS) > 1)
        .vectorize(state.range(VECTOR_SIZE) > 1)
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) == 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .unroll_factor(state.range(UNROLL_FACTOR));
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q3.voila";
    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        // qualification data
        int32_t segment = queryGenerator->getQ3CompressedSegment(*benchmarkState);
        int32_t date = queryGenerator->getQ3Date();
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
        state.SetIterationTime(prog.getExecTime() / 1000);
    }
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
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_shippriority_ref(htSizes, INVALID<int32_t>::val);
        std::vector<double> sum_disc_price_ref(htSizes, 0);
        int32_t segment = queryGenerator->getQ3CompressedSegment(*benchmarkState);
        int32_t date = queryGenerator->getQ3Date();
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
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_shippriority_ref(htSizes, INVALID<int32_t>::val);
        std::vector<double> sum_disc_price_ref(htSizes, 0);
        int32_t segment = queryGenerator->getQ3CompressedSegment(*benchmarkState);
        int32_t date = queryGenerator->getQ3Date();
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

static void Q3_Uncompressed(benchmark::State &state)
{
    using namespace voila;
    auto l_orderkey = benchmarkState->getLineitem().getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto l_extendedprice = benchmarkState->getLineitem().getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = benchmarkState->getLineitem().getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_shipdate = benchmarkState->getLineitem().getColumn<std::string>(L_SHIPDATE);
    auto c_mktsegment = benchmarkState->getCustomer().getColumn<std::string>(C_MKTSEGMENT);
    auto c_custkey = benchmarkState->getCustomer().getColumn<customer_types_t<C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = benchmarkState->getOrders().getColumn<orders_types_t<O_CUSTKEY>>(O_CUSTKEY);
    auto o_orderkey = benchmarkState->getOrders().getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto o_orderdate = benchmarkState->getOrders().getColumn<std::string>(O_ORDERDATE);
    auto o_shippriority = benchmarkState->getOrders().getColumn<orders_types_t<O_SHIPPRIORITY>>(O_SHIPPRIORITY);

    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto date = queryGenerator->getQ3Date();
        auto segment = queryGenerator->getQ3Segment();
        state.ResumeTiming();

        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<std::string> ht_o_orderdate_ref(htSizes, INVALID<std::string>::val);
        std::vector<int32_t> ht_o_shippriority_ref(htSizes, INVALID<int32_t>::val);
        std::vector<double> sum_disc_price_ref(htSizes, 0);

        const auto jt = joins::NPO_st(
            l_orderkey,
            [&l_shipdate, date](auto idx, auto) { return DateReformatter::parseDate(l_shipdate[idx]) > date; },
            o_orderkey,
            [&o_orderdate, date](auto idx, auto) { return DateReformatter::parseDate(o_orderdate[idx]) < date; });
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
            const auto idx = probeAndInsert(
                hash(l_orderkey[l_idx], DateReformatter::parseDate(o_orderdate[o_idx]), o_shippriority[o_idx]),
                l_orderkey[l_idx], o_orderdate[o_idx], o_shippriority[o_idx], ht_l_orderkey_ref, ht_o_orderdate_ref,
                ht_o_shippriority_ref);

            sum_disc_price_ref[idx] += l_extendedprice[l_idx] * (1 - l_discount[l_idx]);
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
        .tile(state.range(TILING) > 0)
        .optimize_selections(state.range(OPTIMIZE_SELECTIONS) > 1)
        .vectorize(state.range(VECTOR_SIZE) > 1)
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) == 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .unroll_factor(state.range(UNROLL_FACTOR));
    config.debug(false).optimize().tile().async_parallel(false).openmp_parallel();

    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q6.voila";

    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        auto startDate = queryGenerator->getQ6Date();
        auto endDate = startDate + 10000;
        auto quantity = queryGenerator->getQ6Quantity();
        auto discount = queryGenerator->getQ6Discount();
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
        state.SetIterationTime(prog.getExecTime() / 1000);
    }
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
        const auto startDate = queryGenerator->getQ6Date();
        const auto endDate = startDate + 10000;
        const auto quantity = queryGenerator->getQ6Quantity();
        const auto discount = queryGenerator->getQ6Discount();
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
        .tile(state.range(TILING) > 0)
        .optimize_selections(state.range(OPTIMIZE_SELECTIONS) > 1)
        .vectorize(state.range(VECTOR_SIZE) > 1)
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) == 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .unroll_factor(state.range(UNROLL_FACTOR));

    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q9.voila";

    Program prog(query, config);
    for ([[maybe_unused]] auto _ : state)
    {
        auto needle = queryGenerator->getQ9CompressedColor(*benchmarkState);
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

        state.SetIterationTime(prog.getExecTime() / 1000);
    }
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
        auto needle = queryGenerator->getQ9CompressedColor(*benchmarkState);
        state.ResumeTiming();
        // reference impl
        // double ref = 0;
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_n_name_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_needle_ref(std::bit_ceil(needle.size()), INVALID<int32_t>::val);
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

static void Q9_Joins(benchmark::State &state)
{
    using namespace voila;
    auto part = benchmarkState->getPartCompressed();
    auto supplier = benchmarkState->getSupplierCompressed();
    auto lineitem = benchmarkState->getLineitemCompressed();
    auto partsupp = benchmarkState->getPartsuppCompressed();
    auto orders = benchmarkState->getOrdersCompressed();
    auto nation = benchmarkState->getNationCompressed();

    auto n_name = nation.getColumn<nation_types_t<N_NAME>>(N_NAME);
    auto o_orderdate = orders.getColumn<orders_types_t<O_ORDERDATE>>(O_ORDERDATE);
    auto l_extendedprice = lineitem.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto ps_supplycost = partsupp.getColumn<partsupp_types_t<PS_SUPPLYCOST>>(PS_SUPPLYCOST);
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto s_suppkey = supplier.getColumn<supplier_types_t<S_SUPPKEY>>(S_SUPPKEY);
    auto l_suppkey = lineitem.getColumn<lineitem_types_t<L_SUPPKEY>>(L_SUPPKEY);
    std::vector<uint64_t> ps_suppkey_partkey;
    ps_suppkey_partkey.reserve(ps_supplycost.size());
    for (const auto &ps : ranges::views::zip(partsupp.getColumn<partsupp_types_t<PS_SUPPKEY>>(PS_SUPPKEY),
                                             partsupp.getColumn<partsupp_types_t<PS_PARTKEY>>(PS_PARTKEY)))
    {
        ps_suppkey_partkey.push_back(static_cast<uint64_t>(ps.first) << 32 | static_cast<uint64_t>(ps.second));
    }
    auto l_partkey = lineitem.getColumn<lineitem_types_t<L_PARTKEY>>(L_PARTKEY);
    auto p_partkey = part.getColumn<part_types_t<P_PARTKEY>>(P_PARTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto s_nationkey = supplier.getColumn<supplier_types_t<S_NATIONKEY>>(S_NATIONKEY);
    auto n_nationkey = nation.getColumn<nation_types_t<N_NATIONKEY>>(N_NATIONKEY);
    auto p_name = part.getColumn<int32_t>(P_NAME);

    // qualification data
    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto needle = queryGenerator->getQ9CompressedColor(*benchmarkState);
        state.ResumeTiming();
        // reference impl
        // double ref = 0;
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_n_name_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_needle_ref(std::bit_ceil(needle.size()), INVALID<int32_t>::val);
        std::vector<double> sum_disc_price_ref(htSizes, 0);

        for (const auto &elem : needle)
        {
            auto h = hash(elem);
            probeAndInsert(h, elem, ht_needle_ref);
        }

        // joins
        const auto part_lineitem = joins::NPO_st(
            p_partkey,
            [&p_name, &ht_needle_ref](auto idx, auto)
            { return contains(hash(p_name[idx]), p_name[idx], ht_needle_ref); },
            l_partkey, [](auto, auto) { return true; });

        // materialize orders
        decltype(l_orderkey) m_orderkey;
        m_orderkey.reserve(part_lineitem.size());
        for (auto &el : part_lineitem)
        {
            m_orderkey.push_back(l_orderkey[el.second]);
        }

        const auto part_lineitem_orders = joins::NPO_st(m_orderkey, o_orderkey);

        // materialize supplier
        decltype(l_orderkey) m_suppkey;
        m_suppkey.reserve(part_lineitem_orders.size());
        for (auto &el : part_lineitem_orders)
        {
            m_suppkey.push_back(l_orderkey[part_lineitem[el.first].second]);
        }

        const auto part_lineitem_orders_supplier = joins::NPO_st(m_suppkey, s_suppkey);

        // materialize ps
        std::vector<uint64_t> m_suppkey_partkey;
        m_suppkey_partkey.reserve(part_lineitem_orders_supplier.size());
        for (auto &el : part_lineitem_orders_supplier)
        {
            m_suppkey_partkey.push_back(
                static_cast<uint64_t>(l_partkey[part_lineitem[part_lineitem_orders[el.first].first].second]) << 32 |
                static_cast<uint64_t>(l_suppkey[part_lineitem[part_lineitem_orders[el.first].first].second]));
        }

        // join lineitem and partsupp
        const auto part_lineitem_orders_supplier_partsupp = joins::NPO_st(m_suppkey_partkey, ps_suppkey_partkey);

        // materialize nationkey
        decltype(s_nationkey) m_nationkey;
        m_suppkey.reserve(part_lineitem_orders_supplier_partsupp.size());
        for (auto &el : part_lineitem_orders_supplier_partsupp)
        {
            m_nationkey.push_back(s_nationkey[part_lineitem_orders_supplier[el.first].second]);
        }

        const auto part_lineitem_orders_supplier_partsupp_nation = joins::NPO_st(m_nationkey, n_nationkey);

        // grouping
        for (const auto &i : part_lineitem_orders_supplier_partsupp_nation)
        {
            const auto nation_idx = i.second;
            const auto order_idx =
                part_lineitem_orders_supplier
                    [part_lineitem_orders_supplier_partsupp[part_lineitem_orders_supplier_partsupp[i.first].first]
                         .first]
                        .first;
            const auto lineitem_idx =
                part_lineitem_orders
                    [part_lineitem_orders_supplier
                         [part_lineitem_orders_supplier_partsupp[part_lineitem_orders_supplier_partsupp[i.first].first]
                              .first]
                             .first]
                        .first;
            const auto partsupp_idx = i.first;
            const auto idx =
                probeAndInsert(hash(n_name[nation_idx], o_orderdate[order_idx] / 10000), n_name[nation_idx],
                               o_orderdate[order_idx] / 10000, ht_n_name_ref, ht_o_orderdate_ref);
            sum_disc_price_ref[idx] += l_extendedprice[lineitem_idx] * (1 - l_discount[lineitem_idx]) -
                                       ps_supplycost[partsupp_idx] * l_quantity[lineitem_idx];
        }
    }
}

static void Q9_Uncompressed(benchmark::State &state)
{
    using namespace voila;
    auto part = benchmarkState->getPart();
    auto supplier = benchmarkState->getSupplier();
    auto lineitem = benchmarkState->getLineitem();
    auto partsupp = benchmarkState->getPartsupp();
    auto orders = benchmarkState->getOrders();
    auto nation = benchmarkState->getNation();

    auto n_name = nation.getColumn<std::string>(N_NAME);
    auto o_orderdate = orders.getColumn<std::string>(O_ORDERDATE);
    auto l_extendedprice = lineitem.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto ps_supplycost = partsupp.getColumn<partsupp_types_t<PS_SUPPLYCOST>>(PS_SUPPLYCOST);
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto s_suppkey = supplier.getColumn<supplier_types_t<S_SUPPKEY>>(S_SUPPKEY);
    auto l_suppkey = lineitem.getColumn<lineitem_types_t<L_SUPPKEY>>(L_SUPPKEY);
    std::vector<uint64_t> ps_suppkey_partkey;
    ps_suppkey_partkey.reserve(ps_supplycost.size());
    for (const auto &ps : ranges::views::zip(partsupp.getColumn<partsupp_types_t<PS_SUPPKEY>>(PS_SUPPKEY),
                                             partsupp.getColumn<partsupp_types_t<PS_PARTKEY>>(PS_PARTKEY)))
    {
        ps_suppkey_partkey.push_back(static_cast<uint64_t>(ps.first) << 32 | static_cast<uint64_t>(ps.second));
    }
    auto l_partkey = lineitem.getColumn<lineitem_types_t<L_PARTKEY>>(L_PARTKEY);
    auto p_partkey = part.getColumn<part_types_t<P_PARTKEY>>(P_PARTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto s_nationkey = supplier.getColumn<supplier_types_t<S_NATIONKEY>>(S_NATIONKEY);
    auto n_nationkey = nation.getColumn<nation_types_t<N_NATIONKEY>>(N_NATIONKEY);
    auto p_name = part.getColumn<std::string>(P_NAME);

    // qualification data
    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto needle = queryGenerator->getQ9Color();
        state.ResumeTiming();
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<std::string> ht_n_name_ref(htSizes, "");
        std::vector<std::string> ht_o_orderdate_ref(htSizes, "");
        std::vector<double> sum_disc_price_ref(htSizes, 0);

        // joins
        const auto part_lineitem = joins::NPO_st(
            p_partkey, [&p_name](auto idx, auto) { return p_name[idx].find("green") != std::string::npos; }, l_partkey,
            [](auto, auto) { return true; });

        // materialize orders
        decltype(l_orderkey) m_orderkey;
        m_orderkey.reserve(part_lineitem.size());
        for (auto &el : part_lineitem)
        {
            m_orderkey.push_back(l_orderkey[el.second]);
        }

        const auto part_lineitem_orders = joins::NPO_st(m_orderkey, o_orderkey);

        // materialize supplier
        decltype(l_orderkey) m_suppkey;
        m_suppkey.reserve(part_lineitem_orders.size());
        for (auto &el : part_lineitem_orders)
        {
            m_suppkey.push_back(l_orderkey[part_lineitem[el.first].second]);
        }

        const auto part_lineitem_orders_supplier = joins::NPO_st(m_suppkey, s_suppkey);

        // materialize ps
        std::vector<uint64_t> m_suppkey_partkey;
        m_suppkey_partkey.reserve(part_lineitem_orders_supplier.size());
        for (auto &el : part_lineitem_orders_supplier)
        {
            m_suppkey_partkey.push_back(
                static_cast<uint64_t>(l_partkey[part_lineitem[part_lineitem_orders[el.first].first].second]) << 32 |
                static_cast<uint64_t>(l_suppkey[part_lineitem[part_lineitem_orders[el.first].first].second]));
        }

        // join lineitem and partsupp
        const auto part_lineitem_orders_supplier_partsupp = joins::NPO_st(m_suppkey_partkey, ps_suppkey_partkey);

        // materialize nationkey
        decltype(s_nationkey) m_nationkey;
        m_suppkey.reserve(part_lineitem_orders_supplier_partsupp.size());
        for (auto &el : part_lineitem_orders_supplier_partsupp)
        {
            m_nationkey.push_back(s_nationkey[part_lineitem_orders_supplier[el.first].second]);
        }

        const auto part_lineitem_orders_supplier_partsupp_nation = joins::NPO_st(m_nationkey, n_nationkey);

        // grouping
        for (const auto &i : part_lineitem_orders_supplier_partsupp_nation)
        {
            const auto nation_idx = i.second;
            const auto order_idx =
                part_lineitem_orders_supplier
                    [part_lineitem_orders_supplier_partsupp[part_lineitem_orders_supplier_partsupp[i.first].first]
                         .first]
                        .first;
            const auto lineitem_idx =
                part_lineitem_orders
                    [part_lineitem_orders_supplier
                         [part_lineitem_orders_supplier_partsupp[part_lineitem_orders_supplier_partsupp[i.first].first]
                              .first]
                             .first]
                        .first;
            const auto partsupp_idx = i.first;
            const auto idx =
                probeAndInsert(hash(DateReformatter::parseDate(o_orderdate[order_idx]) / 10000, n_name[nation_idx]),
                               n_name[nation_idx], o_orderdate[order_idx].substr(4), ht_n_name_ref, ht_o_orderdate_ref);
            benchmark::DoNotOptimize(sum_disc_price_ref[idx] +=
                                     l_extendedprice[lineitem_idx] * (1 - l_discount[lineitem_idx]) -
                                     ps_supplycost[partsupp_idx] * l_quantity[lineitem_idx]);
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
        .tile(state.range(TILING) > 0)
        .optimize_selections(state.range(OPTIMIZE_SELECTIONS) > 1)
        .vectorize(state.range(VECTOR_SIZE) > 1)
        .vector_size(state.range(VECTOR_SIZE))
        .parallelize(state.range(PARALLELIZE_TYPE) == 0)
        .async_parallel(state.range(PARALLELIZE_TYPE) == 1)
        .openmp_parallel(state.range(PARALLELIZE_TYPE) == 2)
        .unroll_factor(state.range(UNROLL_FACTOR));
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q18.voila";
    Program prog(query, config);

    for ([[maybe_unused]] auto _ : state)
    {
        auto quantity = queryGenerator->getQ18Quantity();
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

        state.SetIterationTime(prog.getExecTime() / 1000);
    }

    // state.counters["Query Runtime"] = benchmark::Counter(queryTime, benchmark::Counter::kAvgIterations);
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
        auto quantity = queryGenerator->getQ18Quantity();
        state.ResumeTiming();
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_c_name_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_c_custkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
        std::vector<double> ht_o_totalprice_ref(htSizes, INVALID<int32_t>::val);
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

static void Q18_Joins(benchmark::State &state)
{
    using namespace voila;
    auto customer = benchmarkState->getCustomerCompressed();
    auto orders = benchmarkState->getOrdersCompressed();
    auto lineitem = benchmarkState->getLineitemCompressed();
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto c_name = customer.getColumn<customer_types_t<C_NAME>>(C_NAME);
    auto c_custkey = customer.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = orders.getColumn<orders_types_t<O_CUSTKEY>>(O_CUSTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto o_orderdate = orders.getColumn<orders_types_t<O_ORDERDATE>>(O_ORDERDATE);
    auto o_totalprice = orders.getColumn<orders_types_t<O_TOTALPRICE>>(O_TOTALPRICE);
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);

    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto quantity = queryGenerator->getQ18Quantity();
        state.ResumeTiming();
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_c_name_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_c_custkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
        std::vector<double> ht_o_totalprice_ref(htSizes, INVALID<int32_t>::val);
        std::vector<double> sum_l_quantity(htSizes, 0);
        std::vector<double> sum_l_quantity2(htSizes, 0);

        for (size_t i = 0; i < l_orderkey.size(); ++i)
        {
            const auto idx = probeAndInsert(hash(l_orderkey[i]), l_orderkey[i], ht_l_orderkey_ref);
            sum_l_quantity[idx] += l_quantity[i];
        }

        const auto jt = joins::NPO_st(
            l_orderkey,
            [&sum_l_quantity, &ht_l_orderkey_ref, &quantity](const auto, const auto &key)
            {
                auto pRes = probe(hash(key), key, ht_l_orderkey_ref);
                return pRes && sum_l_quantity[*pRes] > quantity;
            },
            o_orderkey, [](auto, auto) { return true; });

        decltype(o_custkey) m_custkey;
        m_custkey.reserve(jt.size());
        for (auto &el : jt)
        {
            m_custkey.push_back(o_custkey[el.second]);
        }
        const auto jt2 = joins::NPO_st(
            m_custkey, [](auto, auto) { return true; }, c_custkey, [](auto, auto) { return true; });

        for (const auto &el : jt2)
        {
            const auto l_idx = jt[el.first].first;
            const auto o_idx = jt[el.first].second;
            const auto idx = probeAndInsert(hash(c_name[el.second], c_custkey[el.second], o_orderkey[o_idx],
                                                 o_orderdate[o_idx], o_totalprice[o_idx]),
                                            c_name[el.second], c_custkey[el.second], o_orderkey[o_idx],
                                            o_orderdate[o_idx], o_totalprice[o_idx], ht_c_name_ref, ht_c_custkey_ref,
                                            ht_o_orderkey_ref, ht_o_orderdate_ref, ht_o_totalprice_ref);
            sum_l_quantity2[idx] += l_quantity[l_idx];
        }
    }
}

static void Q18_Uncompressed(benchmark::State &state)
{
    using namespace voila;
    auto customer = benchmarkState->getCustomer();
    auto orders = benchmarkState->getOrders();
    auto lineitem = benchmarkState->getLineitem();
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto c_name = customer.getColumn<std::string>(C_NAME);
    auto c_custkey = customer.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = orders.getColumn<orders_types_t<O_CUSTKEY>>(O_CUSTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto o_orderdate = orders.getColumn<std::string>(O_ORDERDATE);
    auto o_totalprice = orders.getColumn<orders_types_t<O_TOTALPRICE>>(O_TOTALPRICE);
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);

    for ([[maybe_unused]] auto _ : state)
    {
        state.PauseTiming();
        auto quantity = queryGenerator->getQ18Quantity();
        state.ResumeTiming();
        const auto htSizes = std::bit_ceil(l_orderkey.size());
        std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<std::string> ht_c_name_ref(htSizes, INVALID<std::string>::val);
        std::vector<int32_t> ht_c_custkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<int32_t> ht_o_orderkey_ref(htSizes, INVALID<int32_t>::val);
        std::vector<std::string> ht_o_orderdate_ref(htSizes, INVALID<std::string>::val);
        std::vector<double> ht_o_totalprice_ref(htSizes, INVALID<int32_t>::val);
        std::vector<double> sum_l_quantity(htSizes, 0);
        std::vector<double> sum_l_quantity2(htSizes, 0);

        for (size_t i = 0; i < l_orderkey.size(); ++i)
        {
            const auto idx = probeAndInsert(hash(l_orderkey[i]), l_orderkey[i], ht_l_orderkey_ref);
            sum_l_quantity[idx] += l_quantity[i];
        }

        const auto jt = joins::NPO_st(
            l_orderkey,
            [&sum_l_quantity, &ht_l_orderkey_ref, &quantity](const auto, const auto &key)
            {
                auto pRes = probe(hash(key), key, ht_l_orderkey_ref);
                return pRes && sum_l_quantity[*pRes] > quantity;
            },
            o_orderkey, [](auto, auto) { return true; });

        decltype(o_custkey) m_custkey;
        m_custkey.reserve(jt.size());
        for (auto &el : jt)
        {
            m_custkey.push_back(o_custkey[el.second]);
        }
        const auto jt2 = joins::NPO_st(
            m_custkey, [](auto, auto) { return true; }, c_custkey, [](auto, auto) { return true; });

        for (const auto &el : jt2)
        {
            const auto l_idx = jt[el.first].first;
            const auto o_idx = jt[el.first].second;
            const auto idx = probeAndInsert(hash(c_name[el.second], c_custkey[el.second], o_orderkey[o_idx],
                                                 o_orderdate[o_idx], o_totalprice[o_idx]),
                                            c_name[el.second], c_custkey[el.second], o_orderkey[o_idx],
                                            o_orderdate[o_idx], o_totalprice[o_idx], ht_c_name_ref, ht_c_custkey_ref,
                                            ht_o_orderkey_ref, ht_o_orderdate_ref, ht_o_totalprice_ref);
            sum_l_quantity2[idx] += l_quantity[l_idx];
        }
    }
}

BENCHMARK(Q1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(10)
    ->ArgsProduct({/*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(1, 16, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency(), 2),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 2, 1),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 1, 1)})
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[(tmp.size() / 4) * 3];
                        });
BENCHMARK(Q3)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(10)
    ->ArgsProduct({/*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(1, 16, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency(), 2),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 2, 1),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 1, 1)})
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[(tmp.size() / 4) * 3];
                        });
BENCHMARK(Q6)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(10)
    ->ArgsProduct({/*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(1, 16, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency(), 2),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 2, 1),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 1, 1)})
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[(tmp.size() / 4) * 3];
                        });
BENCHMARK(Q9)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(10)
    ->ArgsProduct({/*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(1, 16, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency(), 2),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 2, 1),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 1, 1)})
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[(tmp.size() / 4) * 3];
                        });
BENCHMARK(Q18)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(10)
    ->ArgsProduct({/*tiling*/ benchmark::CreateDenseRange(0, 1, 1),
                   /*vectorize*/ benchmark::CreateRange(1, 16, 2),
                   /*unroll*/ benchmark::CreateRange(1, std::thread::hardware_concurrency(), 2),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 2, 1),
                   /*async/openmp*/ benchmark::CreateDenseRange(0, 1, 1)})
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[(tmp.size() / 4) * 3];
                        });

BENCHMARK(Q1_Baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[(tmp.size() / 4) * 3];
                        });
BENCHMARK(Q3_Baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[(tmp.size() / 4) * 3];
                        });
BENCHMARK(Q3_JoinCompressed)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q3_Uncompressed)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q6_Baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q9_Baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q9_Joins)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q9_Uncompressed)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q18_Baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q18_Joins)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });
BENCHMARK(Q18_Uncompressed)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->ComputeStatistics("max",
                        [](const std::vector<double> &v) -> double
                        { return *(std::max_element(std::begin(v), std::end(v))); })
    ->ComputeStatistics("min", [](const std::vector<double> &v) -> double { return *ranges::min_element(v); })
    ->ComputeStatistics("median",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 2);
                            return tmp[tmp.size() / 2];
                        })
    ->ComputeStatistics("lower",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4);
                            return tmp[tmp.size() / 4];
                        })
    ->ComputeStatistics("upper",
                        [](const std::vector<double> &v) -> double
                        {
                            auto tmp = v;
                            ranges::nth_element(tmp, tmp.begin() + tmp.size() / 4 * 3);
                            return tmp[tmp.size() / 4 * 3];
                        });