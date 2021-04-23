#include "ast/Loop.hpp"

namespace voila::ast
{
    std::string Loop::type2string() const
    {
        return "loop";
    }
    Loop *Loop::as_loop()
    {
        return this;
    }
    bool Loop::is_loop() const
    {
        return true;
    }
    void Loop::print(std::ostream &ostream) const
    {
        ostream << "loop";
    }
} // namespace voila::ast