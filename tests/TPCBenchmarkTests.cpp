#include "Config.hpp"
#include "Profiler.hpp"
#include "Program.hpp"
#include "Tables.hpp"
#include "benchmark_defs.hpp.inc"

#include <algorithm>
#include <gtest/gtest.h>
#include <xxhash.h>
using namespace voila;

/*template<size_t ...args>
struct sum
{
    static constexpr auto value = (0 + ... + args);
};

template<class ...Ts>
static size_t hash(Ts... t)
{
    std::array<char, sum<sizeof(Ts)...>::value> data{};
    auto tpl = std::make_tuple(::std::move(t)...);
    tpl.
    return XXH3_64bits(data.data(), sum<sizeof(Ts)...>::value);
}*/

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

constexpr int32_t INVALID = static_cast<int32_t>(std::numeric_limits<uint64_t>::max());
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

TEST(TPCBenchmarkTests, Q1_Qualification)
{
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin.xz");
    auto l_quantity = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(lineitem_cols::L_QUANTITY);
    auto l_extendedprice =
        lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(lineitem_cols::L_DISCOUNT);
    auto l_tax = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_TAX>>(lineitem_cols::L_TAX);
    auto l_returnflag = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_RETURNFLAG>>(lineitem_cols::L_RETURNFLAG);
    auto l_linestatus = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_LINESTATUS>>(lineitem_cols::L_LINESTATUS);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(lineitem_cols::L_SHIPDATE);
    const auto htSizes = std::bit_ceil(l_quantity.size());

    // tpc qualification date
    auto date = 19980901;
    std::vector<int32_t> ht_returnflag_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_linestatus_ref(htSizes, static_cast<int32_t>(INVALID));
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
    auto ref_idx = std::distance(ht_linestatus_ref.begin(), std::find_if(
                                                                ht_linestatus_ref.begin(), ht_linestatus_ref.end(),
                                                                [](auto elem) -> auto { return elem != INVALID; }));

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
    CompressedTable lineitem(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin.xz");
    auto l_quantity = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_QUANTITY>>(lineitem_cols::L_QUANTITY);
    auto l_extendedprice =
        lineitem.getColumn<lineitem_types_t<lineitem_cols::L_EXTENDEDPRICE>>(lineitem_cols::L_EXTENDEDPRICE);
    auto l_discount = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_DISCOUNT>>(lineitem_cols::L_DISCOUNT);
    auto l_shipdate = lineitem.getColumn<lineitem_types_t<lineitem_cols::L_SHIPDATE>>(lineitem_cols::L_SHIPDATE);
    Config config;
    config.debug().tile(false).async_parallel(false).openmp_parallel();
    // config.parallelize=false;
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

    // qualification data
    int32_t segment = customer_orders_lineitem.getDictionary(customer_cols::C_MKTSEGMENT).at("BUILDING");
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
    std::vector<int32_t> ht_l_orderkey_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_o_shippriority_ref(htSizes, static_cast<int32_t>(INVALID));
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

TEST(TPCBenchmarkTests, Q9_Qualification)
{
    // load data
    CompressedTable part_supplier_lineitem_partsupp_orders_nation(VOILA_BENCHMARK_DATA_PATH "/q9_wide_table.bin.xz");
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
    std::vector<int32_t> ht_n_name_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_needle_ref(std::bit_ceil(needle.size()), static_cast<int32_t>(INVALID));
    std::vector<double> sum_disc_price_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();
    assert(std::ranges::all_of(
        needle, [](const auto &e) -> auto { return e != -1 && e != 0; }));
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
    /*    prog << n_name;
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
        prog << p_name;*/

    prog << needle;
    auto find = [](auto iter, auto val)
    {
        for (auto v : iter)
            if (v == val)
                return true;
        return false;
    };
    auto res = prog();
    auto el = *std::get<strided_memref_ptr<uint32_t, 1>>(res[0]);
    for (auto e : needle)
    {
        assert(find(el, (uint32_t) e));
    }

    // TODO: comparisons
    // EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}

TEST(TPCBenchmarkTests, Q18_Qualification)
{
    // load data
    CompressedTable customer_orders_lineitem(VOILA_BENCHMARK_DATA_PATH "/customer_orders_lineitem.bin.xz");
    constexpr auto orders_offset = magic_enum::enum_count<customer_cols>();
    constexpr auto lineitem_offset = orders_offset + magic_enum::enum_count<orders_cols>();
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

    // qualification data
    int32_t quantity = 300;

    // voila calculations
    Config config;
    config.debug().parallelize(false).tile(false);
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
    std::vector<int32_t> ht_l_orderkey_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_o_orderdate_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_o_shippriority_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<double> sum_qty_ref(htSizes, 0);
    std::vector<double> sum_base_price_ref(htSizes, 0);
    std::vector<double> sum_disc_price_ref(htSizes, 0);
    std::vector<double> sum_charge_ref(htSizes, 0);
    std::vector<double> sum_discount_ref(htSizes, 0);
    std::vector<double> avg_qty_ref(htSizes, 0);
    std::vector<double> avg_price_ref(htSizes, 0);
    std::vector<double> avg_disc_ref(htSizes, 0);
    std::vector<double> count_order_ref(htSizes, 0);

    Profiler<Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, /*Events::TLB_MISSES,*/
             Events::NO_INST_COMPLETE, /*Events::CY_STALLED,*/ Events::REF_CYCLES, Events::TOT_CYCLES,
             Events::INS_ISSUED, Events::PREFETCH_MISS>
        prof;
    prof.start();

    /*    for (size_t i = 0; i < l_orderkey.size(); ++i)
        {
            if (c_mktsegment[i] == segment && c_custkey[i] == o_custkey[i] && l_orderkey[i] == o_orderkey[i] &&
                o_orderdate[i] < date && l_shipdate[i] > date)
            {
                const auto idx = probeAndInsert(hash(l_orderkey[i], o_orderdate[i], o_shippriority[i]), htSizes,
                                                l_orderkey[i], o_orderdate[i], o_shippriority[i], ht_l_orderkey_ref,
                                                ht_o_orderdate_ref, ht_o_shippriority_ref);

                sum_disc_price_ref[idx] += l_extendedprice[i] * (1 - l_discount[i]);
            }
        }*/
    prof.stop();
    std::cout << prof << std::endl;

    // TODO: comparisons
    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
}
