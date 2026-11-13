package com.eignex.koblas.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

import static java.lang.constant.ConstantDescs.*;
import static java.lang.constant.DirectMethodHandleDesc.Kind.STATIC;
import static org.junit.jupiter.api.Assertions.*;

public class BenchmarkCallbacksTest {
    private static final ClassFile CLASS_FILE = ClassFile.of();
    private static final ClassDesc OPERATION = ClassDesc.of("com.eignex.koblas.bench.BenchmarkOperation");
    private static final ClassDesc FIXTURE = ClassDesc.of("com.eignex.koblas.bench.CallbackFixture");
    private static final ClassDesc FUNCTION = ClassDesc.of("kotlin.jvm.functions.Function0");
    private static final ClassDesc NUMBER = ClassDesc.of("java.lang.Number");
    private static final ClassDesc BOX = ClassDesc.of("java.lang.Double");
    private static final MethodTypeDesc NUMERIC = MethodTypeDesc.of(CD_double);

    @Test
    void primitiveCallbacksAndUntimedSetupAreAllowed() {
        var classes = fixture(NUMERIC, code -> code.dconst_1().dreturn(), false);
        assertTrue(BenchmarkCallbacks.violations(classes).isEmpty());
    }

    @Test
    void genericInvocationBehindAHelperIsRejected() {
        var classes = fixture(NUMERIC, code -> code.invokestatic(FIXTURE, "helper", NUMERIC).dreturn(), false);
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("Function0.invoke")));
    }

    @Test
    void boxingInsideAPrimitiveCallbackIsRejected() {
        var classes = fixture(NUMERIC, code -> code.dconst_1()
            .invokestatic(BOX, "valueOf", MethodTypeDesc.of(BOX, CD_double))
            .invokevirtual(BOX, "doubleValue", NUMERIC).dreturn(), false);
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("Double.valueOf")));
    }

    @Test
    void boxedMethodReferencesCannotHideBehindSamAdaptation() {
        var classes = fixture(MethodTypeDesc.of(BOX), code -> code.dconst_1()
            .invokestatic(BOX, "valueOf", MethodTypeDesc.of(BOX, CD_double)).areturn(), false);
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("adapts a boxed")));
    }

    @Test
    void classBasedSamImplementationsAreChecked() {
        var classes = fixture(NUMERIC, code -> code.invokestatic(FIXTURE, "helper", NUMERIC).dreturn(), true);
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("Function0.invoke")));
    }

    @Test
    void virtualHelpersCannotHideGenericCallbacks() {
        ClassDesc step = ClassDesc.of("com.eignex.koblas.bench.NumericStep");
        var classes = fixture(NUMERIC, code -> code.aconst_null().invokeinterface(step, "compute", NUMERIC).dreturn(), false);
        classes.put("NumericStep", CLASS_FILE.build(step, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withMethod("compute", NUMERIC, ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {})));
        classes.put("StepImpl", CLASS_FILE.build(ClassDesc.of("com.eignex.koblas.bench.StepImpl"), builder -> builder
            .withInterfaceSymbols(step)
            .withMethodBody("compute", NUMERIC, ClassFile.ACC_PUBLIC,
                code -> code.invokestatic(FIXTURE, "helper", NUMERIC).dreturn())));
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("Function0.invoke")));
    }

    @Test
    void inheritedVirtualHelpersCannotHideGenericCallbacks() {
        ClassDesc step = ClassDesc.of("com.eignex.koblas.bench.NumericStep");
        ClassDesc base = ClassDesc.of("com.eignex.koblas.bench.BaseStep");
        var classes = fixture(NUMERIC, code -> code.aconst_null().invokeinterface(step, "compute", NUMERIC).dreturn(), false);
        classes.put("NumericStep", CLASS_FILE.build(step, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withMethod("compute", NUMERIC, ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {})));
        classes.put("BaseStep", CLASS_FILE.build(base, builder -> builder
            .withMethodBody("compute", NUMERIC, ClassFile.ACC_PUBLIC,
                code -> code.invokestatic(FIXTURE, "helper", NUMERIC).dreturn())));
        classes.put("StepImpl", CLASS_FILE.build(ClassDesc.of("com.eignex.koblas.bench.StepImpl"), builder -> builder
            .withSuperclass(base).withInterfaceSymbols(step)));
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("Function0.invoke")));
    }

    @Test
    void inheritedInterfaceDefaultsCannotHideGenericCallbacks() {
        ClassDesc base = ClassDesc.of("com.eignex.koblas.bench.BaseStep");
        ClassDesc step = ClassDesc.of("com.eignex.koblas.bench.NumericStep");
        var classes = fixture(NUMERIC, code -> code.aconst_null().invokeinterface(step, "compute", NUMERIC).dreturn(), false);
        classes.put("BaseStep", CLASS_FILE.build(base, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withMethodBody("compute", NUMERIC, ClassFile.ACC_PUBLIC,
                code -> code.invokestatic(FIXTURE, "helper", NUMERIC).dreturn())));
        classes.put("NumericStep", CLASS_FILE.build(step, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(base)));
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("Function0.invoke")));
    }

    @Test
    void inheritedSamImplementationsCannotHideGenericCallbacks() {
        var classes = inheritedSam(code -> code.invokestatic(FIXTURE, "helper", NUMERIC).dreturn());
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("Function0.invoke")));
    }

    @Test
    void inheritedPrimitiveSamImplementationsAreAllowed() {
        assertTrue(BenchmarkCallbacks.violations(inheritedSam(code -> code.dconst_1().dreturn())).isEmpty());
    }

    @Test
    void legacyCallbackStorageIsRejected() {
        var classes = fixture(NUMERIC, code -> code.dconst_1().dreturn(), false);
        classes.put("CaseWork", CLASS_FILE.build(ClassDesc.of("com.eignex.koblas.bench.CaseWork"), builder ->
            builder.withMethodBody("getRun", MethodTypeDesc.of(FUNCTION), ClassFile.ACC_PUBLIC,
                code -> code.aconst_null().areturn())));
        assertTrue(BenchmarkCallbacks.violations(classes).stream().anyMatch(message -> message.contains("CaseWork.run")));
    }

    private static Map<String, byte[]> fixture(MethodTypeDesc signature, Consumer<CodeBuilder> body, boolean classBased) {
        var classes = new HashMap<String, byte[]>();
        classes.put("BenchmarkOperation", CLASS_FILE.build(OPERATION, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withMethod("invoke", NUMERIC, ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {})));
        classes.put("CaseWork", CLASS_FILE.build(ClassDesc.of("com.eignex.koblas.bench.CaseWork"), builder ->
            builder.withMethodBody("getRun", MethodTypeDesc.of(OPERATION), ClassFile.ACC_PUBLIC,
                code -> code.aconst_null().areturn())));
        classes.put("CallbackFixture", CLASS_FILE.build(FIXTURE, builder -> {
            if (classBased) builder.withInterfaceSymbols(OPERATION);
            builder.withMethodBody(classBased ? "invoke" : "numeric", signature,
                ClassFile.ACC_PUBLIC | (classBased ? 0 : ClassFile.ACC_STATIC), body);
            builder.withMethodBody("helper", NUMERIC, ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                code -> code.aconst_null().invokeinterface(FUNCTION, "invoke", MethodTypeDesc.of(CD_Object))
                    .checkcast(NUMBER).invokevirtual(NUMBER, "doubleValue", NUMERIC).dreturn());
            builder.withMethodBody("setup", MethodTypeDesc.of(CD_void), ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                code -> code.aconst_null().invokeinterface(FUNCTION, "invoke", MethodTypeDesc.of(CD_Object))
                    .pop().return_());
            if (!classBased) builder.withMethodBody("factory", MethodTypeDesc.of(OPERATION),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, code -> code.invokedynamic(factory(signature)).areturn());
        }));
        return classes;
    }

    private static Map<String, byte[]> inheritedSam(Consumer<CodeBuilder> body) {
        var classes = fixture(NUMERIC, code -> code.dconst_1().dreturn(), false);
        ClassDesc base = ClassDesc.of("com.eignex.koblas.bench.BaseOperation");
        classes.put("BaseOperation", CLASS_FILE.build(base, builder -> builder
            .withMethodBody("invoke", NUMERIC, ClassFile.ACC_PUBLIC, body)));
        classes.put("OperationImpl", CLASS_FILE.build(ClassDesc.of("com.eignex.koblas.bench.OperationImpl"), builder -> builder
            .withSuperclass(base).withInterfaceSymbols(OPERATION)));
        return classes;
    }

    private static DynamicCallSiteDesc factory(MethodTypeDesc signature) {
        ClassDesc lookup = ClassDesc.of("java.lang.invoke.MethodHandles$Lookup");
        ClassDesc type = ClassDesc.of("java.lang.invoke.MethodType");
        ClassDesc handle = ClassDesc.of("java.lang.invoke.MethodHandle");
        var bootstrap = MethodHandleDesc.ofMethod(STATIC, ClassDesc.of("java.lang.invoke.LambdaMetafactory"),
            "metafactory", MethodTypeDesc.of(ClassDesc.of("java.lang.invoke.CallSite"),
                lookup, CD_String, type, type, handle, type));
        return DynamicCallSiteDesc.of(bootstrap, "invoke", MethodTypeDesc.of(OPERATION), NUMERIC,
            MethodHandleDesc.ofMethod(STATIC, FIXTURE, "numeric", signature), NUMERIC);
    }
}
