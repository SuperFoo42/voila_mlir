//===- X86VectorToLLVMIRTranslation.cpp - Translate X86Vector to LLVM IR---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR X86Vector dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialects/FPMath/Translation/FPMathToLLVMIRTranslation.h"
#include "mlir/Dialects/FPMath/IR/FPMathDialect.h"
#include "mlir/Dialects/FPMath/IR/FPMathOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/FixedPointBuilder.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::fpmath;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the X86Vector dialect to LLVM IR.
    class FPMathLLVMIRTranslationInterface
            : public LLVMTranslationDialectInterface {

        static llvm::FixedPointSemantics getFPSemantic(DecimalType type) {
            return {static_cast<unsigned int>(type.getWidth()), static_cast<unsigned int>(type.getScale()), true, false,
                    false};
        }

    public:
        using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

        /// Translates the given operation to LLVM IR using the provided IR builder
        /// and saving the state in `moduleTranslation`.
        LogicalResult
        convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                         LLVM::ModuleTranslation &moduleTranslation) const final {
            Operation &opInst = *op;
            llvm::FixedPointBuilder<llvm::IRBuilderBase> fixedPointBuilder(builder);

            return TypeSwitch<Operation *, LogicalResult>(op)
                    .Case<fpmath::ConstantOp>([&](fpmath::ConstantOp &op) { return failure(); })
                    .Case<fpmath::AddOp>([&](fpmath::AddOp &op) {
                        fixedPointBuilder.CreateAdd(moduleTranslation.lookupValue(op.getLhs()),
                                                    getFPSemantic(op.getLhs().getType()),
                                                    moduleTranslation.lookupValue(op.getRhs()),
                                                    getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::SubOp>([&](fpmath::SubOp &op) {
                        fixedPointBuilder.CreateSub(moduleTranslation.lookupValue(op.getLhs()),
                                                    getFPSemantic(op.getLhs().getType()),
                                                    moduleTranslation.lookupValue(op.getRhs()),
                                                    getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::MulOp>([&](fpmath::MulOp &op) {
                        fixedPointBuilder.CreateMul(moduleTranslation.lookupValue(op.getLhs()),
                                                    getFPSemantic(op.getLhs().getType()),
                                                    moduleTranslation.lookupValue(op.getRhs()),
                                                    getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::DivOp>([&](fpmath::DivOp &op) {
                        fixedPointBuilder.CreateDiv(moduleTranslation.lookupValue(op.getLhs()),
                                                    getFPSemantic(op.getLhs().getType()),
                                                    moduleTranslation.lookupValue(op.getRhs()),
                                                    getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::EqOp>([&](fpmath::EqOp &op) {
                        fixedPointBuilder.CreateEQ(moduleTranslation.lookupValue(op.getLhs()),
                                                   getFPSemantic(op.getLhs().getType()),
                                                   moduleTranslation.lookupValue(op.getRhs()),
                                                   getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::NeqOp>([&](fpmath::NeqOp &op) {
                        fixedPointBuilder.CreateNE(moduleTranslation.lookupValue(op.getLhs()),
                                                   getFPSemantic(op.getLhs().getType()),
                                                   moduleTranslation.lookupValue(op.getRhs()),
                                                   getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::LtOp>([&](fpmath::LtOp &op) {
                        fixedPointBuilder.CreateLT(moduleTranslation.lookupValue(op.getLhs()),
                                                   getFPSemantic(op.getLhs().getType()),
                                                   moduleTranslation.lookupValue(op.getRhs()),
                                                   getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::LeOp>([&](fpmath::LeOp &op) {
                        fixedPointBuilder.CreateLE(moduleTranslation.lookupValue(op.getLhs()),
                                                   getFPSemantic(op.getLhs().getType()),
                                                   moduleTranslation.lookupValue(op.getRhs()),
                                                   getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::GeOp>([&](fpmath::GeOp &op) {
                        fixedPointBuilder.CreateGE(moduleTranslation.lookupValue(op.getLhs()),
                                                   getFPSemantic(op.getLhs().getType()),
                                                   moduleTranslation.lookupValue(op.getRhs()),
                                                   getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::GtOp>([&](fpmath::GtOp &op) {
                        fixedPointBuilder.CreateGT(moduleTranslation.lookupValue(op.getLhs()),
                                                   getFPSemantic(op.getLhs().getType()),
                                                   moduleTranslation.lookupValue(op.getRhs()),
                                                   getFPSemantic(op.getRhs().getType()));
                        return success();
                    })
                    .Case<fpmath::FpToDecOp>([&](fpmath::FpToDecOp &op) {
                        fixedPointBuilder.CreateFloatingToFixed(moduleTranslation.lookupValue(op.getOperand()),
                                                                getFPSemantic(
                                                                        op.getResult().getType().dyn_cast<DecimalType>()));
                        return success();
                    })
                    .Case<fpmath::IntToDecOp>([&](fpmath::IntToDecOp &op) {
                        fixedPointBuilder.CreateIntegerToFixed(
                                moduleTranslation.lookupValue(op.getOperand()), /*isSigned*/ true, //always signed
                                getFPSemantic(op.getResult().getType().dyn_cast<DecimalType>()));
                        return success();
                    })
                    .Case<fpmath::DecToFltOp>([&](fpmath::DecToFltOp &op) {
                        fixedPointBuilder.CreateFixedToFloating(moduleTranslation.lookupValue(op.getArg()),
                                                                getFPSemantic(op.getArg().getType()),
                                                                moduleTranslation.convertType(op->getResultTypes()[0]));
                        return success();
                    })
                    .Case<fpmath::DecToIntOp>([&](fpmath::DecToIntOp &op) {
                        fixedPointBuilder.CreateFixedToInteger(moduleTranslation.lookupValue(op.getArg()),
                                                               getFPSemantic(op.getArg().getType()),
                                                               op->getResultTypes()[0].dyn_cast<IntegerType>().getWidth(),
                                                               true /*always signed*/
                        );
                        return success();
                    })
                    .Default([](Operation *op) { return failure(); });
        }
    };
} // namespace

void mlir::registerFPMathTranslation(DialectRegistry &registry) {
    registry.insert<fpmath::FPMathDialect>();
    registry.addExtension(
            +[](MLIRContext *ctx, fpmath::FPMathDialect *dialect) {
                dialect->addInterfaces<FPMathLLVMIRTranslationInterface>();
            });
}

void mlir::registerFPMathTranslation(MLIRContext &context) {
    DialectRegistry registry;
    registerFPMathTranslation(registry);
    context.appendDialectRegistry(registry);
}
