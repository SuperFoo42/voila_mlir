add_mlir_dialect(FPMathOps fpmath)
add_mlir_doc(FPMathOps FPMathDialect FPMathTypes FPMathAttr fpmath -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS FPMathAttr.td)
mlir_tablegen(FPMathAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(FPMathAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRFPMathAttrIncGen)