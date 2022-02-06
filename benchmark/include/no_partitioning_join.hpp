/**
 * @file    no_partitioning_join.c
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Sun Feb  5 20:16:58 2012
 * @version $Id: no_partitioning_join.c 4419 2013-10-21 16:24:35Z bcagri $
 *
 * @brief  The implementation of NPO, No Partitioning Optimized join algortihm.
 *
 * (c) 2012, ETH Zurich, Systems Group
 *
 */

#include <bit>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <vector>
#include <xxhash.h>

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
// 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

namespace voila::joins
{
    using result_t = std::vector<std::pair<size_t, size_t>>;

    constexpr int32_t HASH(auto X, auto MASK, auto SKIP) requires std::is_scalar_v<decltype(X)>
    {
        return (XXH3_64bits(&X, sizeof(X)) & MASK) >> SKIP;
    }

    constexpr int32_t HASH(auto X, auto MASK, auto SKIP) requires std::is_same_v<decltype(X), std::string>
    {
        return (XXH3_64bits(X.data(), X.size()) & MASK) >> SKIP;
    }

    /** Type definition for a tuple, depending on KEY_8B a tuple can be 16B or 8B */
    template<class T>
    struct Tuple
    {
        size_t key;
        T payload;
    };

    /**
     * Normal hashtable buckets.
     *
     * if KEY_8B then key is 8B and sizeof(Bucket) = 48B
     * else key is 16B and sizeof(Bucket) = 32B
     */

    template<class T, size_t BUCKET_SIZE>
    struct __attribute__((aligned(hardware_destructive_interference_size))) Bucket
    {
        volatile char latch = 0;
        /* 3B hole */
        uint32_t count = 0;
        Tuple<T> tuples[BUCKET_SIZE];
        Bucket *next = nullptr; // TODO: memleak

        ~Bucket() = default;
    };


    /** Hashtable structure for NPO. */
    template<class T, size_t BUCKET_SIZE = 2>
    requires std::is_default_constructible_v<T>
    class HashTable
    {
      public:
        uint32_t num_buckets; // std::hardware_destructive_interference_size,
        uint32_t hash_mask;
        uint32_t skip_bits;
        using bucket_t = Bucket<T, BUCKET_SIZE>;
        using tuple_t = Tuple<T>;

        std::unique_ptr<bucket_t[]> buckets;

        explicit HashTable(uint32_t nbuckets) :
            num_buckets(std::bit_ceil<uint32_t>(nbuckets / BUCKET_SIZE)),
            hash_mask(num_buckets - 1),
            skip_bits(0),
            buckets(new  bucket_t[num_buckets])
        {
            std::fill(reinterpret_cast<char *>(buckets.get()), reinterpret_cast<char *>(buckets.get() + num_buckets),
                      0);
        }

        template<class Predicate>
        void build_hashtable_st(const std::vector<T> &rel, Predicate pred)
        {
            for (const auto &elem : ranges::views::enumerate(rel))
            {
                if (pred(std::get<0>(elem), std::get<1>(elem)))
                {
                    tuple_t *dest;
                    auto idx = HASH(std::get<1>(elem), hash_mask, skip_bits);

                    /* copy the tuple to appropriate hash bucket */
                    /* if full, follow nxt pointer to find correct place */
                    auto &curr = buckets[idx];
                    auto *nxt = curr.next;

                    if (curr.count == BUCKET_SIZE)
                    {
                        if (!nxt || nxt->count == BUCKET_SIZE)
                        {
                            bucket_t *b;
                            b = new bucket_t;
                            curr.next = b;
                            b->next = nxt;
                            b->count = 1;
                            dest = b->tuples;
                        }
                        else
                        {
                            dest = nxt->tuples + nxt->count;
                            nxt->count++;
                        }
                    }
                    else
                    {
                        dest = curr.tuples + curr.count;
                        curr.count++;
                    }
                    *dest = tuple_t{std::get<0>(elem), std::get<1>(elem)};
                }
            }
        }

        void build_hashtable_st(const std::vector<T> &rel)
        {
            for (const auto &elem : ranges::views::enumerate(rel))
            {
                tuple_t *dest;
                auto idx = HASH(std::get<1>(elem), hash_mask, skip_bits);

                /* copy the tuple to appropriate hash bucket */
                /* if full, follow nxt pointer to find correct place */
                auto &curr = buckets[idx];
                auto *nxt = curr.next;

                if (curr.count == BUCKET_SIZE)
                {
                    if (!nxt || nxt->count == BUCKET_SIZE)
                    {
                        bucket_t *b;
                        b = new bucket_t;
                        curr.next = b;
                        b->next = nxt;
                        b->count = 1;
                        dest = b->tuples;
                    }
                    else
                    {
                        dest = nxt->tuples + nxt->count;
                        nxt->count++;
                    }
                }
                else
                {
                    dest = curr.tuples + curr.count;
                    curr.count++;
                }
                tuple_t nT = {std::get<0>(elem), std::get<1>(elem)};
                *dest = nT;
            }
        }

        /**
         * Probes the hashtable for the given outer relation, returns num results.
         * This probing method is used for both single and multi-threaded version.
         *
         * @param ht hashtable to be probed
         * @param rel the probing outer relation
         * @param output chained tuple buffer to write join results, i.e. rid pairs.
         *
         * @return number of matching tuples
         */
        template<class Predicate>
        result_t probe_hashtable(const std::vector<T> &rel, Predicate pred)
        {
            uint32_t j;
            result_t results;

            for (const auto &elem : ranges::views::enumerate(rel))
            {
                if (pred(std::get<0>(elem), std::get<1>(elem)))
                {
                    auto idx = HASH(std::get<1>(elem), hash_mask, skip_bits);
                    bucket_t *b = &buckets[idx];

                    do
                    {
                        for (j = 0; j < b->count; j++)
                        {
                            if (std::get<1>(elem) == b->tuples[j].payload)
                                results.emplace_back(b->tuples[j].key, std::get<0>(elem));
                        }

                        b = b->next; /* follow overflow pointer */
                    } while (b);
                }
            }

            return results;
        }

        result_t probe_hashtable(const std::vector<T> &rel)
        {
            uint32_t j;
            result_t results;

            for (const auto &elem : ranges::views::enumerate(rel))
            {
                auto idx = HASH(std::get<1>(elem), hash_mask, skip_bits);
                bucket_t *b = &buckets[idx];

                do
                {
                    for (j = 0; j < b->count; j++)
                    {
                        if (std::get<1>(elem) == b->tuples[j].payload)
                            results.emplace_back(b->tuples[j].key, std::get<0>(elem));
                    }

                    b = b->next; /* follow overflow pointer */
                } while (b);
            }

            return results;
        }
    };

    /** \copydoc NPO_st */
    template<class T, class SelFuncL, class SelFuncR>
    result_t NPO_st(std::vector<T> &relR, SelFuncL selL, std::vector<T> &relS, SelFuncR selR)
    {
        HashTable<T> ht(relR.size());

        ht.build_hashtable_st(relR, selL);

        return ht.probe_hashtable(relS, selR);
    }

    template<class T>
    result_t NPO_st(std::vector<T> &relR, std::vector<T> &relS)
    {
        HashTable<T> ht(relR.size());

        ht.build_hashtable_st(relR);

        return ht.probe_hashtable(relS);
    }
} // namespace voila::joins