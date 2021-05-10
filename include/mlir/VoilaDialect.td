include "mlir/Interfaces/SideEffectInterfaces.td"

def Voila_Dialect : Dialect {
    let name = "voila";
    let summary = "A high-level dialect for adaptive reprogramming of database queries on heterogeneous systems";
    let description = [{
    TODO - a long description of this dialect...

    }];

    let cppNamespace = "::mlir::voila";
}