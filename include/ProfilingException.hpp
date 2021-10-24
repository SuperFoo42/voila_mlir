#pragma once
#include <stdexcept>

class ProfilingException : public std::runtime_error
{
  public:
    ProfilingException() : std::runtime_error("A profiling error occurred") {}
    explicit ProfilingException(const std::string &error) : std::runtime_error(error) {}
    ~ProfilingException() noexcept override = default;
};

class HardwareLimitProfilingException : public ProfilingException
{
  public:
    explicit HardwareLimitProfilingException() :
        ProfilingException("Not enough resources to add Event to Profiler")
    {
    }
    ~HardwareLimitProfilingException() noexcept override = default;
};