package com.eignex.koblas.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static java.lang.constant.ConstantDescs.*;
import static java.lang.constant.DirectMethodHandleDesc.Kind.STATIC;
import static org.junit.jupiter.api.Assertions.*;

public class KernelBoundariesTest {
    private static final ClassDesc LEAF = ClassDesc.of("com.eignex.koblas.internal.kernels.scalar.dense.VectorKernelsKt");
    private static final ClassDesc DOUBLES = CD_double.arrayType();
    private static final ClassDesc VECTOR = ClassDesc.of("jdk.incubator.vector.DoubleVector");
    private static final ClassDesc VECTOR_HELPER = ClassDesc.of("com.eignex.koblas.internal.VectorHelper");

    @Test
    void survivingArithmeticLeafCallIsRejected() {
        byte[] caller = fixture(code -> code.aconst_null().iconst_0()
            .invokestatic(LEAF, "sum", MethodTypeDesc.of(CD_double, DOUBLES, CD_int)).pop2().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void defaultArgumentBridgeCannotHideALeafCall() {
        byte[] caller = fixture(code -> code.aconst_null().iconst_0().iconst_0().aconst_null()
            .invokestatic(LEAF, "sum$default", MethodTypeDesc.of(CD_double, DOUBLES, CD_int, CD_int, CD_Object))
            .pop2().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void primitiveBackendDispatchRemainsABoundary() {
        ClassDesc backend = ClassDesc.of("com.eignex.koblas.internal.kernels.EngineKernels");
        byte[] caller = fixture(code -> code.aconst_null().aconst_null().iconst_0()
            .invokeinterface(backend, "sum", MethodTypeDesc.of(CD_double, DOUBLES, CD_int)).pop2().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void containerAlgorithmsRemainOutlined() {
        ClassDesc matrix = ClassDesc.of("com.eignex.koblas.SparseMatrix");
        byte[] caller = fixture(code -> code.aconst_null()
            .invokestatic(LEAF, "norm", MethodTypeDesc.of(CD_double, matrix)).pop2().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void vectorArgumentsAndResultsCannotCrossProjectCalls(boolean argument) {
        byte[] caller = fixture(code -> {
            if (argument) code.aconst_null();
            code.invokestatic(VECTOR_HELPER, "vector", argument
                ? MethodTypeDesc.of(CD_double, VECTOR) : MethodTypeDesc.of(VECTOR));
            if (argument) code.pop2(); else code.pop();
            code.return_();
        });
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void jdkVectorIntrinsicsRemainAllowed() {
        ClassDesc species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        byte[] caller = fixture(code -> code.getstatic(VECTOR, "SPECIES_PREFERRED", species)
            .invokestatic(VECTOR, "zero", MethodTypeDesc.of(VECTOR, species)).pop().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void methodHandlesCannotHideALeafCall() {
        var handle = MethodHandleDesc.ofMethod(STATIC, LEAF, "sum", MethodTypeDesc.of(CD_double, DOUBLES));
        byte[] caller = fixture(code -> code.ldc(handle).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void lambdaBootstrapCannotHideALeafCall() {
        ClassDesc lookup = ClassDesc.of("java.lang.invoke.MethodHandles$Lookup");
        ClassDesc methodType = ClassDesc.of("java.lang.invoke.MethodType");
        ClassDesc methodHandle = ClassDesc.of("java.lang.invoke.MethodHandle");
        var bootstrap = MethodHandleDesc.ofMethod(STATIC, ClassDesc.of("java.lang.invoke.LambdaMetafactory"),
            "metafactory", MethodTypeDesc.of(ClassDesc.of("java.lang.invoke.CallSite"),
                lookup, CD_String, methodType, methodType, methodHandle, methodType));
        var signature = MethodTypeDesc.of(CD_double, CD_double);
        var handle = MethodHandleDesc.ofMethod(STATIC, LEAF, "absolute", signature);
        var dynamic = DynamicCallSiteDesc.of(bootstrap, "applyAsDouble",
            MethodTypeDesc.of(ClassDesc.of("java.util.function.DoubleUnaryOperator")), signature, handle, signature);
        byte[] caller = fixture(code -> code.invokedynamic(dynamic).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    private static byte[] fixture(Consumer<CodeBuilder> body) {
        return ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.KernelFixture"), builder ->
            builder.withMethodBody("run", MethodTypeDesc.of(CD_void),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, body));
    }
}
