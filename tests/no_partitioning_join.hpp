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
    constexpr size_t BUCKET_SIZE = 2;

    constexpr int32_t HASH(auto X, auto MASK, auto SKIP)
    {
        return (xxhash(&X, sizeof(X)) & MASK) >> SKIP;
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
    template<class T>
    struct Bucket
    {
        volatile char latch = 0;
        /* 3B hole */
        uint32_t count = 0;
        Tuple<T> tuples[BUCKET_SIZE] = {0, 0};
        Bucket *next = nullptr;
    };

    /** Hashtable structure for NPO. */
    template<class T> requires std::is_trivial_v<T>
    class HashTable
    {
      public:
        uint32_t num_buckets; // std::hardware_destructive_interference_size,
        uint32_t hash_mask;
        uint32_t skip_bits;
        using bucket_t = Bucket<T>;
        using tuple_t = Tuple<T>;
        std::unique_ptr<bucket_t> buckets;

        explicit HashTable(uint32_t nbuckets) :
            num_buckets(std::bit_ceil<uint32_t>(nbuckets)),
            hash_mask(num_buckets - 1),
            skip_bits(0),
            buckets(reinterpret_cast<bucket_t *>(
                std::aligned_alloc(hardware_destructive_interference_size, sizeof(bucket_t) * num_buckets)))
        {
            std::fill(reinterpret_cast<char *>(buckets.get()), reinterpret_cast<char *>(buckets.get() + num_buckets),
                      0);
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
                *dest = elem;
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
        result_t probe_hashtable(const std::vector<T> &rel)
        {
            uint32_t i, j;
            result_t results;

            for (auto elem : rel)
            {
                auto idx = HASH(elem, hash_mask, skip_bits);
                bucket_t *b = buckets[idx];

                do
                {
                    for (j = 0; j < b->count; j++)
                    {
                        if (rel->tuples[i].payload == b->tuples[j].payload)
                            results.push_back(b->tuples[j].key, rel->tuples[i].key);
                    }

                    b = b->next; /* follow overflow pointer */
                } while (b);
            }

            return results;
        }
    };

    /** \copydoc NPO_st */
    template<class T>
    result_t NPO_st(std::vector<T> &relR, std::vector<T> &relS)
    {
        uint32_t nbuckets = (relR.size() / BUCKET_SIZE);
        HashTable<T> ht(nbuckets);

        ht.build_hashtable_st(relR);

        return ht.probe_hashtable(ht, relS);
    }
} // namespace voila::joins