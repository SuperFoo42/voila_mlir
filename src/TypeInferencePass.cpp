#include "TypeInferencePass.hpp"
#include "Program.hpp"     // for Program
#include "TypeInferer.hpp" // for TypeInferer
#include "ast/Fun.hpp"     // for Fun
#include <memory>          // for allocator, __shared_ptr_access, shared_ptr
#include <stdexcept>       // for out_of_range
#include <cstdlib>        // for abort
#include <utility>         // for move

namespace voila
{
    TypeInferer &&TypeInferencePass::inferTypes(Program &prog)
    {
        // visit main function at first, do deduce the othe function signatures
        try
        {
            const auto &main = prog.get_func("main");
            main->visit(inferer);
        }
        catch (const std::out_of_range &ex)
        {
            // TODO: message
            abort();
        }

        /*        for (auto &func : prog.get_funcs())
                {
                    func->visit(inferer);
                }*/

        return std::move(inferer);
    }

} // namespace voila