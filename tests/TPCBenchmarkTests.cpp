#include "Config.hpp"
#include "Program.hpp"
#include "Tables.hpp"
#include "benchmark_defs.hpp.inc"

#include <algorithm>
#include <gtest/gtest.h>
#include <xxhash.h>
using namespace voila;

template<class T1, class T2>
static size_t hash(T1 val1, T2 val2)
{
    std::array<char, sizeof(T1) + sizeof(T2)> data{};
    std::copy_n(reinterpret_cast<char *>(&val1), sizeof(T1), data.data());
    std::copy_n(reinterpret_cast<char *>(&val2), sizeof(T2), data.data() + sizeof(T1));
    return XXH3_64bits(data.data(), sizeof(T1) + sizeof(T2));
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

TEST(TPCBenchmarkTests, Q1_Qualification)
{
    auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_tax = lineitem.getColumn<double>(7);
    auto l_returnflag = lineitem.getColumn<int32_t>(8);
    auto l_linestatus = lineitem.getColumn<int32_t>(9);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);
    const auto htSizes = std::bit_ceil(l_quantity.size());

    // tpc qualification date
    auto date = 19980901;
    std::vector<int32_t> ht_returnflag_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int32_t> ht_linestatus_ref(htSizes, static_cast<int32_t>(INVALID));
    std::vector<int64_t> sum_qty_ref(htSizes, 0);
    std::vector<double> sum_base_price_ref(htSizes, 0);
    std::vector<double> sum_disc_price_ref(htSizes, 0);
    std::vector<double> sum_charge_ref(htSizes, 0);
    std::vector<double> sum_discount_ref(htSizes, 0);
    std::vector<double> avg_qty_ref(htSizes, 0);
    std::vector<double> avg_price_ref(htSizes, 0);
    std::vector<double> avg_disc_ref(htSizes, 0);
    std::vector<int64_t> count_order_ref(htSizes, 0);

    for (size_t i = 0; i < l_quantity.size(); ++i)
    {
        if (l_shipdate[i] <= date)
        {
            const auto idx = probeAndInsert(hash(l_returnflag[i], l_linestatus[i]), htSizes, l_returnflag[i],
                                            l_linestatus[i], ht_returnflag_ref, ht_linestatus_ref);
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
    EXPECT_EQ(ref_idx, 503001);
    EXPECT_EQ(ht_returnflag_ref[ref_idx], 0);
    EXPECT_EQ(ht_linestatus_ref[ref_idx], 0);
    EXPECT_EQ(sum_qty_ref[ref_idx], 37734107);
    EXPECT_DOUBLE_EQ(sum_base_price_ref[ref_idx], 56586554400.729897);
    EXPECT_DOUBLE_EQ(sum_disc_price_ref[ref_idx], 53758257134.865143);
    EXPECT_DOUBLE_EQ(sum_charge_ref[ref_idx], 55909065222.825607);
    EXPECT_DOUBLE_EQ(avg_qty_ref[ref_idx], 25);
    EXPECT_DOUBLE_EQ(avg_price_ref[ref_idx], 38273.129734621602);
    EXPECT_DOUBLE_EQ(avg_disc_ref[ref_idx], 0.04998529583825443);
    EXPECT_EQ(count_order_ref[ref_idx], 1478493);

    Config config;
    config.debug = true;
    config.optimize = true;
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q1.voila";
    Program prog(query, config);

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
    auto ht_returnflag_res = std::get<strided_memref_ptr<uint32_t, 1>>(res[0]);
    size_t res_idx = 0;
    while (ht_returnflag_res->operator[](res_idx) == std::numeric_limits<uint32_t>::max())
    {
        ++res_idx;
    }

    EXPECT_EQ(res_idx, ref_idx);
    auto ht_linestatus_res = std::get<strided_memref_ptr<uint32_t, 1>>(res[1]);
    auto sum_qty_res = std::get<strided_memref_ptr<uint64_t, 1>>(res[2]);
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
    EXPECT_DOUBLE_EQ(sum_charge_res->operator[](res_idx), 55909065222.825607);
    EXPECT_DOUBLE_EQ(avg_qty_res->operator[](res_idx), 25.522005853257337);
    EXPECT_DOUBLE_EQ(avg_price_res->operator[](res_idx), 38273.129734621602);
    EXPECT_DOUBLE_EQ(avg_disc_res->operator[](res_idx), 0.04998529583825443);
    EXPECT_EQ(count_order_res->operator[](res_idx), 1478493);
}
TEST(TPCBenchmarkTests, Q6_Qualification)
{
    auto lineitem = Table::readTable(std::string(VOILA_BENCHMARK_DATA_PATH "/lineitem1g_compressed.bin"));
    auto l_quantity = lineitem.getColumn<int32_t>(4);
    auto l_extendedprice = lineitem.getColumn<double>(5);
    auto l_discount = lineitem.getColumn<double>(6);
    auto l_shipdate = lineitem.getColumn<int32_t>(10);
    Config config;
    config.debug = true;
    config.optimize = true;
    constexpr auto query = VOILA_BENCHMARK_SOURCES_PATH "/Q6.voila";
    Program prog(query, config);

    int32_t startDate = 19940101;
    int32_t endDate = 19950101;
    int32_t quantity = 24;
    double discount = 0.06;
    double minDiscount = discount - 0.01;
    double maxDiscount = discount + 0.01;

    prog << ::voila::make_param(l_quantity.data(), l_quantity.size(), DataType::INT32);
    prog << ::voila::make_param(l_discount.data(), l_discount.size(), DataType::DBL);
    prog << ::voila::make_param(l_shipdate.data(), l_shipdate.size(), DataType::INT32);
    prog << ::voila::make_param(l_extendedprice.data(), l_extendedprice.size(), DataType::DBL);
    prog << ::voila::make_param(&startDate, 0, DataType::INT32);
    prog << ::voila::make_param(&endDate, 0, DataType::INT32);
    prog << ::voila::make_param(&quantity, 0, DataType::INT32);
    prog << ::voila::make_param(&minDiscount, 0, DataType::DBL);
    prog << ::voila::make_param(&maxDiscount, 0, DataType::DBL);
    auto res = std::get<double>(prog()[0]);

    double ref = 0;
    for (size_t i = 0; i < l_quantity.size(); ++i)
    {
        if (l_shipdate[i] >= startDate && l_shipdate[i] < endDate && l_quantity[i] < quantity &&
            l_discount[i] >= minDiscount && l_discount[i] <= maxDiscount)
        {
            ref += l_extendedprice[i] * l_discount[i];
        }
    }

    EXPECT_DOUBLE_EQ(ref, 75293731.05440186); // result is 75293731.05440186, because of float precision errors
    EXPECT_DOUBLE_EQ(res, ref);
}
