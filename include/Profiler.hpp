//TODO Refactor

#pragma once
#include "ProfilingException.hpp"

#include <bitset>
#include <experimental/array>
#include <fmt/format.h>
#include <ostream>
#include <papi.h>
#include <range/v3/view/zip.hpp>
#include <vector>

enum class Events : int
{
    L3_CACHE_ACCESSES = PAPI_L3_TCA,
    L2_CACHE_ACCESSES = PAPI_L2_TCA,
    L3_CACHE_MISSES = PAPI_L3_TCM,
    L2_CACHE_MISSES = PAPI_L2_TCM,
    L1_CACHE_MISSES = PAPI_L1_TCM,
    BRANCH_MISSES = PAPI_BR_MSP,
    TLB_MISSES = PAPI_TLB_DM,
    PREFETCH_MISS = PAPI_PRF_DM,
    CY_STALLED = PAPI_RES_STL,
    REF_CYCLES = PAPI_REF_CYC,
    TOT_CYCLES = PAPI_TOT_CYC,
    INS_ISSUED = PAPI_TOT_INS,
    NO_INST_COMPLETE = PAPI_STL_CCY
};

template<Events... evs>
class Profiler
{
  public:
    Profiler() : eventSet{PAPI_NULL}, cycles{0}, time{0}
    {
        values.fill(0);

        auto retval = PAPI_library_init(PAPI_VER_CURRENT);
        if (retval != PAPI_VER_CURRENT)
        {
            throw ProfilingException(PAPI_strerror(retval));
        }

        if ((retval = PAPI_create_eventset(&eventSet)) != PAPI_OK)
        {
            throw ProfilingException(fmt::format("Error creating profiling event set: {}", PAPI_strerror(retval)));
        }

        try
        {
            tryInitEvents();
        }
        catch (HardwareLimitProfilingException &ex)
        {
            spdlog::debug("Not enough hardware resources, use event multiplexing for profiling.");
            PAPI_cleanup_eventset(eventSet);
            PAPI_destroy_eventset(&eventSet);
            eventSet = PAPI_NULL;

            if ((retval = PAPI_create_eventset(&eventSet)) != PAPI_OK)
            {
                throw ProfilingException(fmt::format("Error creating profiling event set: {}", PAPI_strerror(retval)));
            }

            if ((retval = PAPI_multiplex_init()) != PAPI_OK)
            {
                throw ProfilingException(fmt::format("Error enabling event multiplexing: {}", PAPI_strerror(retval)));
            }

            if ((retval = PAPI_assign_eventset_component(eventSet, 0)) != PAPI_OK)
            {
                throw ProfilingException(
                    fmt::format("Error assigning eventset to component: {}", PAPI_strerror(retval)));
            }

            if ((retval = PAPI_set_multiplex(eventSet)) != PAPI_OK)
            {
                throw ProfilingException(fmt::format("Error enabling event multiplexing: {}", PAPI_strerror(retval)));
            }

            tryInitEvents();
        }
    }

    void tryInitEvents()
    {
        for (const auto &ev : events)
        {
            auto retval = PAPI_add_event(eventSet, static_cast<int>(ev));
            if (retval == PAPI_ECNFLCT)
            {
                throw HardwareLimitProfilingException();
            }
            else if (retval != PAPI_OK)
            {
                throw ProfilingException(
                    fmt::format("Error adding profiling Event {}: {}", eventToString(ev), PAPI_strerror(retval)));
            }
        }
    }

    ~Profiler()
    {
        PAPI_cleanup_eventset(eventSet);
        PAPI_destroy_eventset(&eventSet);
    }

    void start()
    {
        PAPI_start(eventSet);
        cycles = PAPI_get_real_cyc();
        time = PAPI_get_real_usec();
    }

    void stop()
    {
        const auto cyc_tmp = PAPI_get_real_cyc();
        const auto time_tmp = PAPI_get_real_usec();
        PAPI_stop(eventSet, values.data());
        cycles = cyc_tmp - cycles;
        time = time_tmp - time;
    }

    friend std::ostream &operator<<(std::ostream &os, const Profiler &profiler)
    {
        for (auto ev : ranges::zip_view(events, profiler.values))
        {
            os << eventToString(ev.first) << ": " << std::to_string(ev.second) << std::endl;
        }
        os << "TIME : " << std::to_string(profiler.time) << std::endl;
        os << "CYCLES : " << std::to_string(profiler.cycles);
        return os;
    }

    std::vector<std::pair<Events, long long>> result()
    {
        std::vector<std::pair<Events, long long>> vals;
        vals.reserve(events.size());
        for (auto ev : ranges::zip_view(events, values))
        {
            vals.push_back(ev);
        }
        return vals;
    }

  private:
    static constexpr auto eventToString(const Events ev)
    {
        switch (ev)
        {
            case Events::L3_CACHE_ACCESSES:
                return "L3_CACHE_ACCESSES";
            case Events::L2_CACHE_ACCESSES:
                return "L2_CACHE_ACCESSES";
            case Events::L3_CACHE_MISSES:
                return "L3_CACHE_MISSES";
            case Events::L2_CACHE_MISSES:
                return "L2_CACHE_MISSES";
            case Events::L1_CACHE_MISSES:
                return "L1_CACHE_MISSES";
            case Events::BRANCH_MISSES:
                return "BRANCH_MISSES";
            case Events::TLB_MISSES:
                return "TLB_MISSES";
            case Events::PREFETCH_MISS:
                return "PREFETCH_MISS";
            case Events::CY_STALLED:
                return "CY_STALLED";
            case Events::REF_CYCLES:
                return "REF_CYCLES";
            case Events::TOT_CYCLES:
                return "TOT_CYCLES";
            case Events::INS_ISSUED:
                return "INS_ISSUED";
            case Events::NO_INST_COMPLETE:
                return "NO_INST_COMPLETE";
        }
    }

    int eventSet;
    long long cycles;

  public:
    [[nodiscard]] long long int getCycles() const
    {
        return cycles;
    }
    [[nodiscard]] long long getTime() const
    {
        return time/1000;
    }

  private:
    long long time;
    constexpr static auto events = std::experimental::make_array(evs...);
    std::array<long long, sizeof...(evs)> values;
};
