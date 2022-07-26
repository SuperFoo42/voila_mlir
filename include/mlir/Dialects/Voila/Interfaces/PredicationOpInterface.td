#ifndef VOILA_PREDICATION_INTERFACE
#define VOILA_PREDICATION_INTERFACE
include "mlir/IR/OpBase.td"

def PredicationOpInterface : OpInterface<"PredicationOpInterface"> {
  let description = [{
    Interface for Voila Operations that can be predicated with selections
  }];
  let cppNamespace = "::mlir::voila";

  let methods = [
    InterfaceMethod<"TODO","void", "predicate", (ins "Value":$pred)>,
    InterfaceMethod<"TODO","::mlir::Value", "predicated">,
  ];
}


#endif // VOILA_PREDICATION_INTERFACE