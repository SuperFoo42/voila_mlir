#ifndef VOILA_DIALECT
#define VOILA_DIALECT

include "mlir/IR/OpBase.td"

def Voila_Dialect : Dialect {
    let name = "voila";
    let summary = "A high-level dialect for adaptive reprogramming of database queries on heterogeneous systems";
    let description = [{
    TODO - a long description of this dialect...

    }];

    let cppNamespace = "::mlir::voila";
}

//base operation definition
class Voila_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Voila_Dialect, mnemonic, traits>;

#endif // VOILA_DIALECT