#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/lowering/BinaryOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::voila
{
    class EqOp;
    class NeqOp;
    class LeOp;
    class LeqOp;
    class GeqOp;
    class GeOp;
} // namespace mlir::voila

namespace
{
    template <class VCmpOp> struct CmpPred
    {
    };

    template <> struct CmpPred<::mlir::voila::EqOp>
    {
        static const auto value = ::mlir::arith::CmpFPredicate::OEQ;
    };
    template <> struct CmpPred<::mlir::voila::NeqOp>
    {
        static const auto value = ::mlir::arith::CmpFPredicate::ONE;
    };
    template <> struct CmpPred<::mlir::voila::LeOp>
    {
        static const auto value = ::mlir::arith::CmpFPredicate::OLT;
    };
    template <> struct CmpPred<::mlir::voila::LeqOp>
    {
        static const auto value = ::mlir::arith::CmpFPredicate::OLE;
    };
    template <> struct CmpPred<::mlir::voila::GeqOp>
    {
        static const auto value = ::mlir::arith::CmpFPredicate::OGE;
    };
    template <> struct CmpPred<::mlir::voila::GeOp>
    {
        static const auto value = ::mlir::arith::CmpFPredicate::OGT;
    };
    template <class VCmpOp> struct CmpIPred
    {
    };

    template <> struct CmpIPred<::mlir::voila::EqOp>
    {
        static const auto value = ::mlir::arith::CmpIPredicate::eq;
    };
    template <> struct CmpIPred<::mlir::voila::NeqOp>
    {
        static const auto value = ::mlir::arith::CmpIPredicate::ne;
    };
    template <> struct CmpIPred<::mlir::voila::LeOp>
    {
        static const auto value = ::mlir::arith::CmpIPredicate::slt;
    };
    template <> struct CmpIPred<::mlir::voila::LeqOp>
    {
        static const auto value = ::mlir::arith::CmpIPredicate::sle;
    };
    template <> struct CmpIPred<::mlir::voila::GeqOp>
    {
        static const auto value = ::mlir::arith::CmpIPredicate::sge;
    };
    template <> struct CmpIPred<::mlir::voila::GeOp>
    {
        static const auto value = ::mlir::arith::CmpIPredicate::sgt;
    };

    template <class VCmpOp> inline constexpr ::mlir::arith::CmpFPredicate cmp_pred_v = CmpPred<VCmpOp>::value;

    template <class VCmpOp> inline constexpr ::mlir::arith::CmpIPredicate cmp_i_pred_v = CmpIPred<VCmpOp>::value;
} // namespace

namespace voila::mlir::lowering
{
    template <typename T>
    concept CmpOp = requires(T a) {
        std::is_same_v<T, ::mlir::voila::EqOp> || std::is_same_v<T, ::mlir::voila::NeqOp> ||
            std::is_same_v<T, ::mlir::voila::GeqOp> || std::is_same_v<T, ::mlir::voila::GeOp> ||
            std::is_same_v<T, ::mlir::voila::LeqOp> || std::is_same_v<T, ::mlir::voila::LeOp>;
    };
    template <CmpOp op> struct OpConverter<op>
    {
        static constexpr auto int_pred = cmp_i_pred_v<op>;
        static constexpr auto flt_pred = cmp_pred_v<op>;
        static ::mlir::Value createOp(::mlir::ConversionPatternRewriter &rewriter,
                                      ::mlir::Location loc,
                                      ::mlir::Value lhs,
                                      ::mlir::Value rhs)
        {
            if (isFloat(lhs))
                return rewriter.create<::mlir::arith::CmpFOp>(loc, flt_pred, lhs, rhs);
            else
                return rewriter.create<::mlir::arith::CmpIOp>(loc, int_pred, lhs, rhs);
        };
    };

    using EqOpLowering = BinaryOpLowering<::mlir::voila::EqOp>;
    using NeqOpLowering = BinaryOpLowering<::mlir::voila::NeqOp>;
    using LeOpLowering = BinaryOpLowering<::mlir::voila::LeOp>;
    using LeqOpLowering = BinaryOpLowering<::mlir::voila::LeqOp>;
    using GeOpLowering = BinaryOpLowering<::mlir::voila::GeOp>;
    using GeqOpLowering = BinaryOpLowering<::mlir::voila::GeqOp>;
} // namespace voila::mlir::lowering