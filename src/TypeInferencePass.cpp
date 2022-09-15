
#include "TypeInferencePass.hpp"
namespace voila
{
    using namespace voila::ast;

    TypeInferer &&TypeInferencePass::inferTypes(Program &prog) {
        //visit main function at first, do deduce the othe function signatures
        try
        {
            const auto &main = prog.get_func("main");
            main->visit(inferer);
        }
        catch (const std::out_of_range &ex)
        {
            //TODO: message
            abort();
        }

/*        for (auto &func : prog.get_funcs())
        {
            func->visit(inferer);
        }*/

        return std::move(inferer);
    }

} // namespace voila