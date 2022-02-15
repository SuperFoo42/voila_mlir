#include "Config.hpp"
#include "Profiler.hpp"
#include "Program.hpp"
#include "TableReader.hpp"
#include "Tables.hpp"
#include "no_partitioning_join.hpp"

#include <gtest/gtest.h>

#include "BenchmarkUtils.hpp"
using namespace voila;

TEST(TPCBenchmarkTests, Q1_Qualification)
{
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem.bin.xz");
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto l_extendedprice = lineitem.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_tax = lineitem.getColumn<lineitem_types_t<L_TAX>>(L_TAX);
    auto l_returnflag = lineitem.getColumn<lineitem_types_t<L_RETURNFLAG>>(L_RETURNFLAG);
    auto l_linestatus = lineitem.getColumn<lineitem_types_t<L_LINESTATUS>>(L_LINESTATUS);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<L_SHIPDATE>>(L_SHIPDATE);
    const auto htSizes = std::bit_ceil(l_quantity.size());

    // tpc qualification date
    auto date = 19980901;
    std::vector<int32_t> ht_returnflag_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_linestatus_ref(htSizes, INVALID<int32_t>::val);
    std::vector<double> sum_qty_ref(htSizes, 0);
    std::vector<double> sum_base_price_ref(htSizes, 0);
    std::vector<double> sum_disc_price_ref(htSizes, 0);
    std::vector<double> sum_charge_ref(htSizes, 0);
    std::vector<double> sum_discount_ref(htSizes, 0);
    std::vector<double> avg_qty_ref(htSizes, 0);
    std::vector<double> avg_price_ref(htSizes, 0);
    std::vector<double> avg_disc_ref(htSizes, 0);
    std::vector<double> count_order_ref(htSizes, 0);

    for (size_t i = 0; i < l_quantity.size(); ++i)
    {
        if (l_shipdate[i] <= date)
        {
            const auto idx = probeAndInsert(hash(l_returnflag[i], l_linestatus[i]), l_returnflag[i], l_linestatus[i],
                                            ht_returnflag_ref, ht_linestatus_ref);
            sum_qty_ref[idx] += l_quantity[i];
            sum_base_price_ref[idx] += l_extendedprice[i];
            sum_disc_price_ref[idx] += l_extendedprice[i] * (1 - l_discount[i]);
            sum_charge_ref[idx] += l_extendedprice[i] * (1 - l_discount[i]) * (1 + l_tax[i]);
            ++count_order_ref[idx];
            avg_qty_ref[idx] = sum_qty_ref[idx] / count_order_ref[idx];
            avg_price_ref[idx] = sum_base_price_ref[idx] / count_order_ref[idx];
            sum_discount_ref[idx] += l_discount[i];
            avg_disc_ref[idx] = sum_discount_ref[idx] / count_order_ref[idx];
        }
    }
    auto ref_idx =
        std::distance(ht_linestatus_ref.begin(), std::find_if(
                                                     ht_linestatus_ref.begin(), ht_linestatus_ref.end(),
                                                     [](auto elem) -> auto { return elem != INVALID<int32_t>::val; }));

    // results slightly differ from sample output, because of float precision errors
    EXPECT_EQ(ref_idx, 46);
    EXPECT_EQ(ht_returnflag_ref[ref_idx], 1);
    EXPECT_EQ(ht_linestatus_ref[ref_idx], 1);
    EXPECT_EQ(sum_qty_ref[ref_idx], 9.914170e+05);
    EXPECT_DOUBLE_EQ(sum_base_price_ref[ref_idx], 1487504710.3799965);
    EXPECT_DOUBLE_EQ(sum_disc_price_ref[ref_idx], 1413082168.0541041);
    EXPECT_DOUBLE_EQ(sum_charge_ref[ref_idx], 1469649223.1943603);
    EXPECT_DOUBLE_EQ(avg_qty_ref[ref_idx], 25.516471920522985);
    EXPECT_DOUBLE_EQ(avg_price_ref[ref_idx], 38284.467760848216);
    EXPECT_DOUBLE_EQ(avg_disc_ref[ref_idx], 0.050093426674193239);
    EXPECT_EQ(count_order_ref[ref_idx], 3.885400e+04);

    Config config;
    config.debug().tile(false);
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q1.voila";
    Program prog(query, config);

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
    auto ht_returnflag_res = std::get<strided_memref_ptr<uint32_t, 1>>(res[0]);
    int64_t res_idx = 0;
    while (ht_returnflag_res->operator[](res_idx) == std::numeric_limits<uint32_t>::max())
    {
        ++res_idx;
    }

    EXPECT_EQ(res_idx, ref_idx);
    auto ht_linestatus_res = std::get<strided_memref_ptr<uint32_t, 1>>(res[1]);
    auto sum_qty_res = std::get<strided_memref_ptr<double, 1>>(res[2]);
    auto sum_base_price_res = std::get<strided_memref_ptr<double, 1>>(res[3]);
    auto sum_disc_price_res = std::get<strided_memref_ptr<double, 1>>(res[4]);
    auto sum_charge_res = std::get<strided_memref_ptr<double, 1>>(res[5]);
    auto avg_price_res = std::get<strided_memref_ptr<double, 1>>(res[6]);
    auto avg_qty_res = std::get<strided_memref_ptr<double, 1>>(res[7]);
    auto avg_disc_res = std::get<strided_memref_ptr<double, 1>>(res[8]);
    auto count_order_res = std::get<strided_memref_ptr<uint64_t, 1>>(res[9]);
    EXPECT_EQ(ht_returnflag_res->operator[](res_idx), 0);
    EXPECT_EQ(ht_linestatus_res->operator[](res_idx), 0);
    EXPECT_EQ(sum_qty_res->operator[](res_idx), 37734107);
    EXPECT_DOUBLE_EQ(sum_base_price_res->operator[](res_idx), 56586554400.729897);
    EXPECT_DOUBLE_EQ(sum_disc_price_res->operator[](res_idx), 53758257134.865143);
    EXPECT_DOUBLE_EQ(sum_charge_res->operator[](res_idx), 55895670983.629333);
    EXPECT_DOUBLE_EQ(avg_qty_res->operator[](res_idx), 25.522005853257337);
    EXPECT_DOUBLE_EQ(avg_price_res->operator[](res_idx), 38273.129734621602);
    EXPECT_DOUBLE_EQ(avg_disc_res->operator[](res_idx), 0.04998529583825443);
    EXPECT_EQ(count_order_res->operator[](res_idx), 1478493);
}

TEST(TPCBenchmarkTests, Q6_Qualification)
{
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem.bin.xz");
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);
    auto l_extendedprice = lineitem.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<L_SHIPDATE>>(L_SHIPDATE);
    Config config;
    config.debug(true).optimize().tile(false).async_parallel(false).openmp_parallel().optimize_selections(false);

    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q6.voila";
    Program prog(query, config);

    int32_t startDate = 19940101;
    int32_t endDate = 19950101;
    int32_t quantity = 24;
    double discount = 0.06;
    double minDiscount = discount - 0.01;
    double maxDiscount = discount + 0.01;

    prog << l_quantity;
    prog << l_discount;
    prog << l_shipdate;
    prog << l_extendedprice;
    prog << &startDate;
    prog << &endDate;
    prog << &quantity;
    prog << &minDiscount;
    prog << &maxDiscount;
    auto res = std::get<double>(prog()[0]);

    double ref = 0;
    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();
    for (size_t i = 0; i < l_quantity.size(); ++i)
    {
        if (l_shipdate[i] >= startDate && l_shipdate[i] < endDate && l_quantity[i] < quantity &&
            l_discount[i] >= minDiscount && l_discount[i] <= maxDiscount)
        {
            ref += l_extendedprice[i] * l_discount[i];
        }
    }
    prof.stop();
    std::cout << prof << std::endl;

    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
    EXPECT_DOUBLE_EQ(res, ref);
}

TEST(TPCBenchmarkTests, Q3_Qualification)
{
    // load data
    CompressedTable customer_orders_lineitem(VOILA_BENCHMARK_DATA_PATH "/customer_orders_lineitem.bin.xz");
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = magic_enum::enum_count<customer_cols>() + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = customer_orders_lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(lineitem_offset + L_ORDERKEY);
    auto l_extendedprice =
        customer_orders_lineitem.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(lineitem_offset + L_EXTENDEDPRICE);
    auto l_discount = customer_orders_lineitem.getColumn<lineitem_types_t<L_DISCOUNT>>(lineitem_offset + L_DISCOUNT);
    auto l_shipdate = customer_orders_lineitem.getColumn<lineitem_types_t<L_SHIPDATE>>(lineitem_offset + L_SHIPDATE);
    auto c_mktsegment = customer_orders_lineitem.getColumn<customer_types_t<C_MKTSEGMENT>>(C_MKTSEGMENT);
    auto c_custkey = customer_orders_lineitem.getColumn<customer_types_t<C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = customer_orders_lineitem.getColumn<orders_types_t<O_CUSTKEY>>(orders_offset + O_CUSTKEY);
    auto o_orderkey = customer_orders_lineitem.getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto o_orderdate = customer_orders_lineitem.getColumn<orders_types_t<O_ORDERDATE>>(orders_offset + O_ORDERDATE);
    auto o_shippriority =
        customer_orders_lineitem.getColumn<orders_types_t<O_SHIPPRIORITY>>(orders_offset + O_SHIPPRIORITY);

    // qualification data
    int32_t segment = customer_orders_lineitem.getDictionary(C_MKTSEGMENT).at("BUILDING");
    int32_t date = 19950315;

    // voila calculations
    Config config;
    config.debug().tile(false).parallelize(false);
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q3.voila";
    Program prog(query, config);
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

    // reference impl
    double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_shippriority_ref(htSizes, INVALID<int32_t>::val);
    std::vector<double> sum_disc_price_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

    for (size_t i = 0; i < l_orderkey.size(); ++i)
    {
        if (c_mktsegment[i] == segment && c_custkey[i] == o_custkey[i] && l_orderkey[i] == o_orderkey[i] &&
            o_orderdate[i] < date && l_shipdate[i] > date)
        {
            const auto idx =
                probeAndInsert(hash(l_orderkey[i], o_orderdate[i], o_shippriority[i]), l_orderkey[i], o_orderdate[i],
                               o_shippriority[i], ht_l_orderkey_ref, ht_o_orderdate_ref, ht_o_shippriority_ref);

            sum_disc_price_ref[idx] += l_extendedprice[i] * (1 - l_discount[i]);
        }
    }
    prof.stop();
    std::cout << prof << std::endl;
    // TODO: comparisons
    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}

TEST(TPCBenchmarkTests, Q3_JoinCompressed)
{
    // load data
    CompressedTable customer(VOILA_BENCHMARK_DATA_PATH "/customer.bin.xz");
    CompressedTable orders(VOILA_BENCHMARK_DATA_PATH "/orders.bin.xz");
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem.bin.xz");
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto l_extendedprice = lineitem.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<L_SHIPDATE>>(L_SHIPDATE);
    auto c_mktsegment = customer.getColumn<customer_types_t<C_MKTSEGMENT>>(C_MKTSEGMENT);
    auto c_custkey = customer.getColumn<customer_types_t<C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = orders.getColumn<orders_types_t<O_CUSTKEY>>(O_CUSTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto o_orderdate = orders.getColumn<orders_types_t<O_ORDERDATE>>(O_ORDERDATE);
    auto o_shippriority = orders.getColumn<orders_types_t<O_SHIPPRIORITY>>(O_SHIPPRIORITY);

    // qualification data
    int32_t segment = customer.getDictionary(customer_cols::C_MKTSEGMENT).at("BUILDING");
    int32_t date = 19950315;

    // reference impl
    double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_shippriority_ref(htSizes, INVALID<int32_t>::val);
    std::vector<double> sum_disc_price_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

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
                                        l_orderkey[l_idx], o_orderdate[o_idx], o_shippriority[o_idx], ht_l_orderkey_ref,
                                        ht_o_orderdate_ref, ht_o_shippriority_ref);

        sum_disc_price_ref[idx] += l_extendedprice[l_idx] * (1 - l_discount[l_idx]);
    }
    prof.stop();
    std::cout << prof << std::endl;
    // TODO: comparisons
    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}

TEST(TPCBenchmarkTests, Q3_JoinUncompressed)
{
    // load data
    auto customer = TableReader(VOILA_BENCHMARK_DATA_PATH "/customer.tbl",
                                {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT,
                                 ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::STRING, ColumnTypes::STRING},
                                '|')
                        .getTable();
    auto orders =
        TableReader(VOILA_BENCHMARK_DATA_PATH "/orders.tbl",
                    {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::DATE,
                     ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT, ColumnTypes::STRING},
                    '|')
            .getTable();
    auto lineitem = TableReader(VOILA_BENCHMARK_DATA_PATH "/lineitem.tbl",
                                {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT,
                                 ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::DECIMAL,
                                 ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::DATE, ColumnTypes::DATE,
                                 ColumnTypes::DATE, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::STRING},
                                '|')
                        .getTable();
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto l_extendedprice = lineitem.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<L_DISCOUNT>>(L_DISCOUNT);
    auto l_shipdate = lineitem.getColumn<std::string>(L_SHIPDATE);
    auto c_mktsegment = customer.getColumn<std::string>(C_MKTSEGMENT);
    auto c_custkey = customer.getColumn<customer_types_t<C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = orders.getColumn<orders_types_t<O_CUSTKEY>>(O_CUSTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto o_orderdate = orders.getColumn<std::string>(O_ORDERDATE);
    auto o_shippriority = orders.getColumn<orders_types_t<O_SHIPPRIORITY>>(O_SHIPPRIORITY);

    // qualification data
    const auto segment = "BUILDING";
    const auto date = 19950315;

    // reference impl
    double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<std::string> ht_o_orderdate_ref(htSizes, INVALID<std::string>::val);
    std::vector<int32_t> ht_o_shippriority_ref(htSizes, INVALID<int32_t>::val);
    std::vector<double> sum_disc_price_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

    const auto jt = joins::NPO_st(
        l_orderkey, [&l_shipdate, date](auto idx, auto) { return DateReformatter::parseDate(l_shipdate[idx]) > date; },
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
    prof.stop();
    std::cout << prof << std::endl;
    // TODO: comparisons
    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}

TEST(TPCBenchmarkTests, Q9_Qualification)
{
    // load data
    CompressedTable part_supplier_lineitem_partsupp_orders_nation(
        VOILA_BENCHMARK_DATA_PATH "/part_supplier_lineitem_partsupp_orders_nation.bin.xz");
    constexpr auto supplier_offset = magic_enum::enum_count<part_cols>();
    constexpr auto lineitem_offset = supplier_offset + magic_enum::enum_count<supplier_cols>();
    constexpr auto partsupp_offset = lineitem_offset + magic_enum::enum_count<lineitem_cols>();
    constexpr auto orders_offset = partsupp_offset + magic_enum::enum_count<partsupp_cols>();
    constexpr auto nations_offset = orders_offset + magic_enum::enum_count<orders_cols>();
    auto n_name =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<nation_types_t<N_NAME>>(nations_offset + N_NAME);
    auto o_orderdate = part_supplier_lineitem_partsupp_orders_nation.getColumn<orders_types_t<O_ORDERDATE>>(
        orders_offset + O_ORDERDATE);
    auto l_extendedprice = part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<L_EXTENDEDPRICE>>(
        lineitem_offset + L_EXTENDEDPRICE);
    auto l_discount = part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<L_DISCOUNT>>(
        lineitem_offset + L_DISCOUNT);
    auto ps_supplycost = part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<PS_SUPPLYCOST>>(
        partsupp_offset + PS_SUPPLYCOST);
    auto l_quantity = part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<L_QUANTITY>>(
        lineitem_offset + L_QUANTITY);
    auto s_suppkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<supplier_types_t<S_SUPPKEY>>(
        supplier_offset + S_SUPPKEY);
    auto l_suppkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<L_SUPPKEY>>(
        lineitem_offset + L_SUPPKEY);
    auto ps_suppkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<PS_SUPPKEY>>(
        partsupp_offset + PS_SUPPKEY);
    auto ps_partkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<partsupp_types_t<PS_PARTKEY>>(
        partsupp_offset + PS_PARTKEY);
    auto l_partkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<L_PARTKEY>>(
        lineitem_offset + L_PARTKEY);
    auto p_partkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<part_types_t<P_PARTKEY>>(P_PARTKEY);
    auto o_orderkey =
        part_supplier_lineitem_partsupp_orders_nation.getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto l_orderkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<lineitem_types_t<L_ORDERKEY>>(
        lineitem_offset + L_ORDERKEY);
    auto s_nationkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<supplier_types_t<S_NATIONKEY>>(
        supplier_offset + S_NATIONKEY);
    auto n_nationkey = part_supplier_lineitem_partsupp_orders_nation.getColumn<nation_types_t<N_NATIONKEY>>(
        nations_offset + N_NATIONKEY);
    auto p_name = part_supplier_lineitem_partsupp_orders_nation.getColumn<int32_t>(P_NAME);

    // qualification data
    std::vector<int32_t> needle;
    for (const auto &entry : part_supplier_lineitem_partsupp_orders_nation.getDictionary(part_cols::P_NAME))
    {
        if (entry.first.find("green") != std::string::npos)
        {
            needle.push_back(entry.second);
        }
    }

    // reference impl
    // double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<int32_t> ht_n_name_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_needle_ref(std::bit_ceil(needle.size()), INVALID<int32_t>::val);
    std::vector<double> sum_disc_price_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

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
            const auto idx = probeAndInsert(hash(n_name[i], o_orderdate[i] / 10000), n_name[i], o_orderdate[i] / 10000,
                                            ht_n_name_ref, ht_o_orderdate_ref);
            sum_disc_price_ref[idx] += l_extendedprice[i] * (1 - l_discount[i]) - ps_supplycost[i] * l_quantity[i];
        }
    }
    prof.stop();
    std::cout << prof << std::endl;

    // voila calculations
    Config config;
    config.optimize(false).debug();
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q9.voila";
    Program prog(query, config);
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
}

TEST(TPCBenchmarkTests, Q9_JoinCompressed)
{
    // load data
    CompressedTable part(VOILA_BENCHMARK_DATA_PATH "/part.bin.xz");
    CompressedTable supplier(VOILA_BENCHMARK_DATA_PATH "/supplier.bin.xz");
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem.bin.xz");
    CompressedTable partsupp(VOILA_BENCHMARK_DATA_PATH "/partsupp.bin.xz");
    CompressedTable orders(VOILA_BENCHMARK_DATA_PATH "/orders.bin.xz");
    CompressedTable nation(VOILA_BENCHMARK_DATA_PATH "/nation.bin.xz");

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
    std::vector<int32_t> needle;
    for (const auto &entry : part.getDictionary(part_cols::P_NAME))
    {
        if (entry.first.find("green") != std::string::npos)
        {
            needle.push_back(entry.second);
        }
    }

    // reference impl
    // double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<int32_t> ht_n_name_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_needle_ref(std::bit_ceil(needle.size()), INVALID<int32_t>::val);
    std::vector<double> sum_disc_price_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

    for (const auto &elem : needle)
    {
        auto h = hash(elem);
        probeAndInsert(h, elem, ht_needle_ref);
    }

    // joins
    const auto part_lineitem = joins::NPO_st(
        p_partkey,
        [&p_name, &ht_needle_ref](auto idx, auto) { return contains(hash(p_name[idx]), p_name[idx], ht_needle_ref); },
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
                [part_lineitem_orders_supplier_partsupp[part_lineitem_orders_supplier_partsupp[i.first].first].first]
                    .first;
        const auto lineitem_idx =
            part_lineitem_orders
                [part_lineitem_orders_supplier
                     [part_lineitem_orders_supplier_partsupp[part_lineitem_orders_supplier_partsupp[i.first].first]
                          .first]
                         .first]
                    .first;
        const auto partsupp_idx = i.first;
        const auto idx = probeAndInsert(hash(n_name[nation_idx], o_orderdate[order_idx] / 10000), n_name[nation_idx],
                                        o_orderdate[order_idx] / 10000, ht_n_name_ref, ht_o_orderdate_ref);
        sum_disc_price_ref[idx] += l_extendedprice[lineitem_idx] * (1 - l_discount[lineitem_idx]) -
                                   ps_supplycost[partsupp_idx] * l_quantity[lineitem_idx];
    }
    prof.stop();
    std::cout << prof << std::endl;
}

TEST(TPCBenchmarkTests, Q9_JoinUncompressed)
{
    // load data
    auto part = TableReader(VOILA_BENCHMARK_DATA_PATH "/part.tbl",
                            {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::STRING,
                             ColumnTypes::STRING, ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::DECIMAL,
                             ColumnTypes::STRING},
                            '|')
                    .getTable();
    auto supplier = TableReader(VOILA_BENCHMARK_DATA_PATH "/supplier.tbl",
                                {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT,
                                 ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::STRING},
                                '|')
                        .getTable();
    auto lineitem = TableReader(VOILA_BENCHMARK_DATA_PATH "/lineitem.tbl",
                                {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT,
                                 ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::DECIMAL,
                                 ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::DATE, ColumnTypes::DATE,
                                 ColumnTypes::DATE, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::STRING},
                                '|')
                        .getTable();
    auto partsupp =
        TableReader(VOILA_BENCHMARK_DATA_PATH "/partsupp.tbl",
                    {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::DECIMAL, ColumnTypes::STRING},
                    '|')
            .getTable();
    auto orders =
        TableReader(VOILA_BENCHMARK_DATA_PATH "/orders.tbl",
                    {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::DATE,
                     ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT, ColumnTypes::STRING},
                    '|')
            .getTable();
    auto nation = TableReader(VOILA_BENCHMARK_DATA_PATH "/nation.tbl",
                              {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::INT, ColumnTypes::STRING}, '|')
                      .getTable();

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

    // reference impl
    // double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<std::string> ht_n_name_ref(htSizes, "");
    std::vector<std::string> ht_o_orderdate_ref(htSizes, "");
    std::vector<double> sum_disc_price_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

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
                [part_lineitem_orders_supplier_partsupp[part_lineitem_orders_supplier_partsupp[i.first].first].first]
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
        sum_disc_price_ref[idx] += l_extendedprice[lineitem_idx] * (1 - l_discount[lineitem_idx]) -
                                   ps_supplycost[partsupp_idx] * l_quantity[lineitem_idx];
    }
    prof.stop();
    std::cout << prof << std::endl;
}

TEST(TPCBenchmarkTests, Q18_Qualification)
{
    // load data
    CompressedTable customer_orders_lineitem(VOILA_BENCHMARK_DATA_PATH "/customer_orders_lineitem.bin.xz");
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = orders_offset + magic_enum::enum_count<orders_cols>();
    auto l_orderkey = customer_orders_lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(lineitem_offset + L_ORDERKEY);
    auto c_name = customer_orders_lineitem.getColumn<customer_types_t<C_NAME>>(C_NAME);
    auto c_custkey = customer_orders_lineitem.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = customer_orders_lineitem.getColumn<orders_types_t<O_CUSTKEY>>(orders_offset + O_CUSTKEY);
    auto o_orderkey = customer_orders_lineitem.getColumn<orders_types_t<O_ORDERKEY>>(orders_offset + O_ORDERKEY);
    auto o_orderdate = customer_orders_lineitem.getColumn<orders_types_t<O_ORDERDATE>>(orders_offset + O_ORDERDATE);
    auto o_totalprice = customer_orders_lineitem.getColumn<orders_types_t<O_TOTALPRICE>>(orders_offset + O_TOTALPRICE);
    auto l_quantity = customer_orders_lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(lineitem_offset + L_QUANTITY);

    // qualification data
    int32_t quantity = 300;

    // voila calculations
    Config config;
    config.debug().optimize(false);
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q18.voila";
    Program prog(query, config);
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
    // reference impl
    double ref = 0;
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

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

    for (size_t i = 0; i < l_orderkey.size(); ++i)
    {
        const auto idx = probeAndInsert(hash(l_orderkey[i]), l_orderkey[i], ht_l_orderkey_ref);
        sum_l_quantity[idx] += l_quantity[i];
        orderkey_idxs[i] = idx;
    }

    for (size_t i = 0; i < l_orderkey.size(); ++i)
    {
        if (sum_l_quantity[orderkey_idxs[i]] > quantity && c_custkey[i] == o_custkey[i] &&
            o_orderkey[i] == l_orderkey[i])
        {
            const auto idx =
                probeAndInsert(hash(c_name[i], c_custkey[i], o_orderkey[i], o_orderdate[i], o_totalprice[i]), c_name[i],
                               c_custkey[i], o_orderkey[i], o_orderdate[i], o_totalprice[i], ht_c_name_ref,
                               ht_c_custkey_ref, ht_o_orderkey_ref, ht_o_orderdate_ref, ht_o_totalprice_ref);
            sum_l_quantity2[idx] += l_quantity[i];
        }
    }

    prof.stop();
    std::cout << prof << std::endl;

    // TODO: comparisons
    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}

TEST(TPCBenchmarkTests, Q18_Compressed)
{
    // load data
    CompressedTable customer(VOILA_BENCHMARK_DATA_PATH "/customer.bin.xz");
    CompressedTable orders(VOILA_BENCHMARK_DATA_PATH "/orders.bin.xz");
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem.bin.xz");
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto c_name = customer.getColumn<customer_types_t<C_NAME>>(C_NAME);
    auto c_custkey = customer.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = orders.getColumn<orders_types_t<O_CUSTKEY>>(O_CUSTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto o_orderdate = orders.getColumn<orders_types_t<O_ORDERDATE>>(O_ORDERDATE);
    auto o_totalprice = orders.getColumn<orders_types_t<O_TOTALPRICE>>(O_TOTALPRICE);
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);

    // qualification data
    int32_t quantity = 300;

    // reference impl
    double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_c_name_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_c_custkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_orderkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, INVALID<int32_t>::val);
    std::vector<double> ht_o_totalprice_ref(htSizes, INVALID<int32_t>::val);
    std::vector<double> sum_l_quantity(htSizes, 0);
    std::vector<double> sum_l_quantity2(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

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
        const auto idx = probeAndInsert(
            hash(c_name[el.second], c_custkey[el.second], o_orderkey[o_idx], o_orderdate[o_idx], o_totalprice[o_idx]),
            c_name[el.second], c_custkey[el.second], o_orderkey[o_idx], o_orderdate[o_idx], o_totalprice[o_idx],
            ht_c_name_ref, ht_c_custkey_ref, ht_o_orderkey_ref, ht_o_orderdate_ref, ht_o_totalprice_ref);
        sum_l_quantity2[idx] += l_quantity[l_idx];
    }

    prof.stop();
    std::cout << prof << std::endl;

    // TODO: comparisons
    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}

TEST(TPCBenchmarkTests, Q18_Uncompressed)
{
    // load data
    auto customer = TableReader(VOILA_BENCHMARK_DATA_PATH "/customer.tbl",
                                {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT,
                                 ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::STRING, ColumnTypes::STRING},
                                '|')
                        .getTable();
    auto orders =
        TableReader(VOILA_BENCHMARK_DATA_PATH "/orders.tbl",
                    {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::DATE,
                     ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT, ColumnTypes::STRING},
                    '|')
            .getTable();
    auto lineitem = TableReader(VOILA_BENCHMARK_DATA_PATH "/lineitem.tbl",
                                {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT,
                                 ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::DECIMAL,
                                 ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::DATE, ColumnTypes::DATE,
                                 ColumnTypes::DATE, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::STRING},
                                '|')
                        .getTable();
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto c_name = customer.getColumn<std::string>(C_NAME);
    auto c_custkey = customer.getColumn<customer_types_t<customer_cols::C_CUSTKEY>>(C_CUSTKEY);
    auto o_custkey = orders.getColumn<orders_types_t<O_CUSTKEY>>(O_CUSTKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto o_orderdate = orders.getColumn<std::string>(O_ORDERDATE);
    auto o_totalprice = orders.getColumn<orders_types_t<O_TOTALPRICE>>(O_TOTALPRICE);
    auto l_quantity = lineitem.getColumn<lineitem_types_t<L_QUANTITY>>(L_QUANTITY);

    // qualification data
    int32_t quantity = 300;

    // reference impl
    double ref = 0;
    const auto htSizes = std::bit_ceil(l_orderkey.size());
    std::vector<int32_t> ht_l_orderkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<std::string> ht_c_name_ref(htSizes, INVALID<std::string>::val);
    std::vector<int32_t> ht_c_custkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<int32_t> ht_o_orderkey_ref(htSizes, INVALID<int32_t>::val);
    std::vector<std::string> ht_o_orderdate_ref(htSizes, INVALID<std::string>::val);
    std::vector<double> ht_o_totalprice_ref(htSizes, INVALID<int32_t>::val);
    std::vector<double> sum_l_quantity(htSizes, 0);
    std::vector<double> sum_l_quantity2(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

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
        const auto idx = probeAndInsert(
            hash(c_name[el.second], c_custkey[el.second], o_orderkey[o_idx], o_orderdate[o_idx], o_totalprice[o_idx]),
            c_name[el.second], c_custkey[el.second], o_orderkey[o_idx], o_orderdate[o_idx], o_totalprice[o_idx],
            ht_c_name_ref, ht_c_custkey_ref, ht_o_orderkey_ref, ht_o_orderdate_ref, ht_o_totalprice_ref);
        sum_l_quantity2[idx] += l_quantity[l_idx];
    }

    prof.stop();
    std::cout << prof << std::endl;

    // TODO: comparisons
    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}

TEST(TPCBenchmarkTests, SimpleJoin)
{
    std::vector<int32_t> t1{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int32_t> t2{2, 4, 6, 8, 1, 1};
    std::vector<std::pair<size_t, size_t>> ref{std::make_pair(1, 0), std::make_pair(3, 1), std::make_pair(5, 2),
                                               std::make_pair(7, 3), std::make_pair(0, 4), std::make_pair(0, 5)};
    auto res = joins::NPO_st(t1, t2);
    ASSERT_EQ(res, ref);
}

TEST(TPCBenchmarkTests, JoinWithPreds)
{
    std::vector<int32_t> t1{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int32_t> t2{2, 4, 6, 8, 1, 1};
    std::vector<std::pair<size_t, size_t>> ref{std::make_pair(1, 0), std::make_pair(3, 1), std::make_pair(5, 2),
                                               std::make_pair(7, 3)};
    auto res = joins::NPO_st(
        t1, [](auto, auto v) { return v % 2 == 0; }, t2, [](auto, auto v) { return v % 2 == 0; });

    ASSERT_EQ(res, ref);
}

TEST(TPCBenchmarkTests, StringJoin)
{
    std::vector<std::string> t1{std::string("1"), std::string("2"), std::string("3"),
                                std::string("4"), std::string("5"), std::string("6"),
                                std::string("7"), std::string("8"), std::string("9")};
    std::vector<std::string> t2{std::string("2"), std::string("4"), std::string("6"),
                                std::string("8"), std::string("1"), std::string("1")};
    std::vector<std::pair<size_t, size_t>> ref{std::make_pair(1, 0), std::make_pair(3, 1), std::make_pair(5, 2),
                                               std::make_pair(7, 3), std::make_pair(0, 4), std::make_pair(0, 5)};
    auto res = joins::NPO_st(t1, t2);
    ASSERT_EQ(res, ref);
}

TEST(TPCBenchmarkTests, LineitemOrdersJoin)
{
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem.bin.xz");
    CompressedTable orders(VOILA_BENCHMARK_DATA_PATH "/orders.bin.xz");
    auto l_orderkey = lineitem.getColumn<lineitem_types_t<L_ORDERKEY>>(L_ORDERKEY);
    auto o_orderkey = orders.getColumn<orders_types_t<O_ORDERKEY>>(O_ORDERKEY);
    auto res = joins::NPO_st(l_orderkey, o_orderkey);
    ASSERT_FALSE(res.empty());
}
