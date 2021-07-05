
#include "TypeInferencePass.hpp"
namespace voila
{
    using namespace voila::ast;

    TypeInferer &&TypeInferencePass::inferTypes(Program &prog) {
        for (auto &func : prog.get_funcs())
        {
            func->visit(inferer);
        }

        return std::move(inferer);
    }

} // namespace voila