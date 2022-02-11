#pragma once
#include <xxhash.h>
#include <cstdint>
#include <array>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <optional>

#include "benchmark_defs.hpp.inc"

template<class T>
static size_t hash(T val1)
{
    if constexpr (std::is_same_v<T, std::string>)
    {
        return XXH3_64bits(val1.data(), val1.size());
    }
    else
        return XXH3_64bits(&val1, sizeof(T));
}
template<class T1, class T2>
static size_t hash(T1 val1, T2 val2)
{
    constexpr auto has_string = std::disjunction_v<std::is_same<T1, std::string>, std::is_same<T2, std::string>>;
    size_t size = 0;
    if constexpr (std::is_same_v<T1, std::string>)
        size += val1.size();
    else
        size += sizeof(T1);
    if constexpr (std::is_same_v<T2, std::string>)
        size += val2.size();
    else
        size += sizeof(T2);

    std::conditional_t<has_string, std::vector<char>, std::array<char, sizeof(T1) + sizeof(T2)>> data{};
    if constexpr (has_string)
        data.resize(size);
    size_t cursor = 0;
    if constexpr (std::is_same_v<T1, std::string>)
    {
        std::copy_n(val1.data(), val1.size(), data.data());
        cursor += val1.size();
    }
    else
    {
        std::copy_n(reinterpret_cast<char *>(&val1), sizeof(T1), data.data());
        cursor += sizeof(T1);
    }
    if constexpr (std::is_same_v<T2, std::string>)
    {
        std::copy_n(val2.data(), val2.size(), data.data() + cursor);
    }
    else
    {
        std::copy_n(reinterpret_cast<char *>(&val2), sizeof(T2), data.data() + cursor);
    }
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
    constexpr auto has_string =
        std::disjunction_v<std::is_same<T1, std::string>, std::is_same<T2, std::string>, std::is_same<T3, std::string>,
                           std::is_same<T4, std::string>, std::is_same<T5, std::string>>;
    size_t size = 0;
    if constexpr (std::is_same_v<T1, std::string>)
        size += val1.size();
    else
        size += sizeof(T1);
    if constexpr (std::is_same_v<T2, std::string>)
        size += val2.size();
    else
        size += sizeof(T2);
    if constexpr (std::is_same_v<T3, std::string>)
        size += val3.size();
    else
        size += sizeof(T3);
    if constexpr (std::is_same_v<T4, std::string>)
        size += val4.size();
    else
        size += sizeof(T4);
    if constexpr (std::is_same_v<T5, std::string>)
        size += val5.size();
    else
        size += sizeof(T5);

    std::conditional_t<has_string, std::vector<char>,
                       std::array<char, sizeof(T1) + sizeof(T2) + sizeof(T3) + sizeof(T4) + sizeof(T5)>>
        data{};
    if constexpr (has_string)
        data.resize(size);
    size_t cursor = 0;
    if constexpr (std::is_same_v<T1, std::string>)
    {
        std::copy_n(val1.data(), val1.size(), data.data());
        cursor += val1.size();
    }
    else
    {
        std::copy_n(reinterpret_cast<char *>(&val1), sizeof(T1), data.data());
        cursor += sizeof(T1);
    }
    if constexpr (std::is_same_v<T2, std::string>)
    {
        std::copy_n(val2.data(), val2.size(), data.data() + cursor);
        cursor += val2.size();
    }
    else
    {
        std::copy_n(reinterpret_cast<char *>(&val2), sizeof(T2), data.data() + cursor);
        cursor += sizeof(T2);
    }
    if constexpr (std::is_same_v<T3, std::string>)
    {
        std::copy_n(val3.data(), val3.size(), data.data() + cursor);
        cursor += val3.size();
    }
    else
    {
        std::copy_n(reinterpret_cast<char *>(&val3), sizeof(T3), data.data() + cursor);
        cursor += sizeof(T3);
    }
    if constexpr (std::is_same_v<T4, std::string>)
    {
        std::copy_n(val4.data(), val4.size(), data.data() + cursor);
        cursor += val4.size();
    }
    else
    {
        std::copy_n(reinterpret_cast<char *>(&val4), sizeof(T4), data.data() + cursor);
        cursor += sizeof(T4);
    }
    if constexpr (std::is_same_v<T5, std::string>)
    {
        std::copy_n(val5.data(), val5.size(), data.data() + cursor);
        cursor += val5.size();
    }
    else
    {
        std::copy_n(reinterpret_cast<char *>(&val5), sizeof(T5), data.data() + cursor);
        cursor += sizeof(T5);
    }

    return XXH3_64bits(data.data(), data.size());
}
template<class T>
struct INVALID
{
    static T val;
};

template<>
struct INVALID<int32_t>
{
    constexpr static int32_t val = static_cast<int32_t>(std::numeric_limits<uint64_t>::max());
};
template<>
struct INVALID<uint64_t>
{
    constexpr static uint64_t val = std::numeric_limits<uint64_t>::max();
};

template<>
struct INVALID<std::string>
{
    constexpr static char val[] = "";
};

template<class T1, class T2>
static size_t probeAndInsert(size_t key, const T1 val1, const T2 val2, std::vector<T1> &vals1, std::vector<T2> &vals2)
{
    assert(vals1.size() == vals2.size());
    const auto size = vals1.size();
    key %= size;
    // probing
    while (vals1[key] != INVALID<T1>::val && !(vals1[key] == val1 && vals2[key] == val2))
    {
        key += 1;
        key %= size;
    }

    vals1[key] = val1;
    vals2[key] = val2;

    return key;
}

template<class T>
static bool contains(size_t key, T val, std::vector<T> &vals)
{
    const auto size = vals.size();
    key %= size;
    // probing
    while (vals[key % size] != INVALID<T>::val && vals[key % size] != val)
    {
        key += 1;
        key %= size;
    }

    return vals[key % size] == val;
}

template<class T>
static std::optional<size_t> probe(size_t key, T val, std::vector<T> &vals)
{
    const auto size = vals.size();
    key %= size;
    // probing
    while (vals[key % size] != INVALID<T>::val && vals[key % size] != val)
    {
        key += 1;
        key %= size;
    }

    return vals[key % size] == val ? std::make_optional(key % size) : std::nullopt;
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
    while (vals1[key] != INVALID<T1>::val && !(vals1[key] == val1 && vals2[key] == val2 && vals3[key] == val3))
    {
        key += 1;
        key %= size;
    }

    vals1[key] = val1;
    vals2[key] = val2;
    vals3[key] = val3;

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
    while (vals1[key] != INVALID<T1>::val && !(vals1[key] == val1 && vals2[key] == val2 && vals3[key] == val3 &&
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

template<class T1>
static size_t probeAndInsert(size_t key, const T1 val1, std::vector<T1> &vals1)
{
    key &= vals1.size() - 1;
    // probing
    while (vals1[key] != INVALID<T1>::val && !(vals1[key] == val1))
    {
        key += 1;
        key &= vals1.size() - 1;
    }

    vals1[key] = val1;

    return key;
}