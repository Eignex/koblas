package com.eignex.koblas.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.DynamicConstantDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.function.Consumer;
import java.util.List;
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

    private static final ClassDesc DENSE_SCHEDULE =
        ClassDesc.of("com.eignex.koblas.internal.scheduling.dense.WindowFixture");
    private static final ClassDesc MATRIX = ClassDesc.of("com.eignex.koblas.DenseMatrix");

    @ParameterizedTest
    @ValueSource(strings = {"wrap", "wrap$default"})
    void denseSchedulesRejectTemporaryMatrixFactories(String name) {
        ClassDesc companion = ClassDesc.of("com.eignex.koblas.DenseMatrix$Companion");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.aconst_null()
            .invokevirtual(companion, name, MethodTypeDesc.of(MATRIX)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void denseSchedulesRejectDirectMatrixConstruction() {
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.new_(MATRIX).dup()
            .invokespecial(MATRIX, "<init>", MethodTypeDesc.of(CD_void)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"view", "view$default"})
    void denseSchedulesRejectTemporaryMatrixViews(String name) {
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.aconst_null()
            .invokevirtual(MATRIX, name, MethodTypeDesc.of(MATRIX)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"DenseVector", "DenseMatrix", "SparseVector", "SparseMatrix"})
    void sparseSchedulesRejectContainerFactories(String name) {
        ClassDesc result = ClassDesc.of("com.eignex.koblas." + name);
        ClassDesc companion = ClassDesc.of("com.eignex.koblas." + name + "$Companion");
        byte[] caller = fixture(ClassDesc.of("com.eignex.koblas.internal.scheduling.sparse.WindowFixture"),
            code -> code.aconst_null().invokevirtual(companion, "wrap", MethodTypeDesc.of(result)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"Vector", "Matrix", "DenseVector", "SparseMatrix"})
    void schedulesRejectContainerResultsFromOrdinaryHelpers(String name) {
        ClassDesc result = ClassDesc.of("com.eignex.koblas." + name);
        ClassDesc helper = ClassDesc.of("com.eignex.koblas.internal.operations.StagingKt");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code
            .invokestatic(helper, "snapshot", MethodTypeDesc.of(result)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesRejectContainerArrays() {
        ClassDesc helper = ClassDesc.of("com.eignex.koblas.internal.operations.StagingKt");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code
            .invokestatic(helper, "snapshots", MethodTypeDesc.of(MATRIX.arrayType())).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesRejectContainerFactoryMethodHandles() {
        ClassDesc helper = ClassDesc.of("com.eignex.koblas.internal.operations.StagingKt");
        var factory = MethodHandleDesc.ofMethod(STATIC, helper, "snapshot", MethodTypeDesc.of(MATRIX, DOUBLES));
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.ldc(factory).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesRejectExternalContainerFactories() {
        ClassDesc helper = ClassDesc.of("external.WindowFactory");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code
            .invokestatic(helper, "snapshot", MethodTypeDesc.of(MATRIX)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesRejectExternalContainerFactoryMethodHandles() {
        ClassDesc helper = ClassDesc.of("external.WindowFactory");
        var factory = MethodHandleDesc.ofMethod(STATIC, helper, "snapshot", MethodTypeDesc.of(MATRIX));
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.ldc(factory).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesRejectExternalPublicEngineHelpers() {
        ClassDesc engine = ClassDesc.of("com.eignex.koblas.KoblasEngine");
        ClassDesc helper = ClassDesc.of("external.Preparation");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.aconst_null()
            .invokestatic(helper, "prepare", MethodTypeDesc.of(CD_void, engine)).return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void scheduleDeclarationsRejectTypedEnginesWithoutCallSites() {
        ClassDesc engine = ClassDesc.of("com.eignex.koblas.KoblasEngine");
        byte[] declaration = ClassFile.of().build(DENSE_SCHEDULE, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT)
            .withMethod("product", MethodTypeDesc.of(CD_void, engine),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void schedulesRejectUnwrappingPublicEnginesThroughHelpers() {
        ClassDesc engine = ClassDesc.of("com.eignex.koblas.KoblasEngine");
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        ClassDesc helper = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProviderKt");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.aconst_null()
            .invokestatic(helper, "getKernels", MethodTypeDesc.of(provider, engine)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesRejectPublicEngineMethodHandles() {
        ClassDesc engine = ClassDesc.of("com.eignex.koblas.KoblasEngine");
        ClassDesc helper = ClassDesc.of("com.eignex.koblas.internal.operations.PreparationKt");
        var call = MethodHandleDesc.ofMethod(STATIC, helper, "run", MethodTypeDesc.of(CD_void, engine));
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.ldc(call).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesMayCallPrimitiveProviders() {
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.aconst_null().aconst_null().iconst_0()
            .invokeinterface(provider, "sum", MethodTypeDesc.of(CD_double, DOUBLES, CD_int)).pop2().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void schedulesRejectReadingContainerGeometry() {
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.aconst_null()
            .invokevirtual(MATRIX, "getRows", MethodTypeDesc.of(CD_int)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void scheduleDeclarationsRejectTypedOperandsWithoutCallSites() {
        byte[] declaration = ClassFile.of().build(DENSE_SCHEDULE, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT)
            .withMethod("product", MethodTypeDesc.of(CD_void, MATRIX),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void schedulesMayValidatePrimitiveExtents() {
        ClassDesc companion = ClassDesc.of("com.eignex.koblas.DenseMatrix$Companion");
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.aconst_null().iconst_0().iconst_0()
            .invokevirtual(companion, "entryCount", MethodTypeDesc.of(CD_int, CD_int, CD_int)).pop().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void schedulesRejectTypedEngineImplementations() {
        ClassDesc engine = ClassDesc.of("com.eignex.koblas.internal.CustomPublicEngine");
        byte[] implementation = ClassFile.of().build(engine, builder -> builder
            .withInterfaceSymbols(ClassDesc.of("com.eignex.koblas.KoblasEngine")));
        byte[] declaration = ClassFile.of().build(DENSE_SCHEDULE, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT)
            .withMethod("product", MethodTypeDesc.of(CD_void, engine),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(List.of(implementation, declaration)).size());
    }

    @Test
    void schedulesRejectInheritedTypedEnginesWithoutOwnMethods() {
        byte[] declaration = ClassFile.of().build(DENSE_SCHEDULE, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(ClassDesc.of("com.eignex.koblas.KoblasEngine")));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"DenseVector", "SparseMatrix", "KoblasEngine"})
    void schedulesRejectStoredTypedOperands(String name) {
        ClassDesc type = ClassDesc.of("com.eignex.koblas." + name).arrayType();
        byte[] declaration = ClassFile.of().build(DENSE_SCHEDULE, builder -> builder
            .withField("operand", type, ClassFile.ACC_PRIVATE | ClassFile.ACC_STATIC));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"DenseMatrix", "KoblasEngine"})
    void schedulesRejectTypedDynamicCallSitesWithGenericBootstraps(String name) {
        ClassDesc lookup = ClassDesc.of("java.lang.invoke.MethodHandles$Lookup");
        ClassDesc methodType = ClassDesc.of("java.lang.invoke.MethodType");
        var bootstrap = MethodHandleDesc.ofMethod(STATIC, ClassDesc.of("external.Bootstrap"), "callSite",
            MethodTypeDesc.of(ClassDesc.of("java.lang.invoke.CallSite"), lookup, CD_String, methodType));
        var dynamic = DynamicCallSiteDesc.of(bootstrap, "operand", MethodTypeDesc.of(ClassDesc.of("com.eignex.koblas." + name)));
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.invokedynamic(dynamic).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"DenseMatrix", "KoblasEngine"})
    void schedulesRejectTypedDynamicConstantsWithGenericBootstraps(String name) {
        var dynamic = typedConstant(ClassDesc.of("com.eignex.koblas." + name));
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.ldc(dynamic).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void schedulesRejectConstructionOfPackagedContainerImplementations() {
        ClassDesc container = ClassDesc.of("com.eignex.koblas.internal.InternalMatrix");
        byte[] implementation = ClassFile.of().build(container, builder -> builder
            .withInterfaceSymbols(ClassDesc.of("com.eignex.koblas.Matrix")));
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.new_(container).dup()
            .invokespecial(container, "<init>", MethodTypeDesc.of(CD_void)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(List.of(implementation, caller)).size());
    }

    @Test
    void backendDeclarationsRejectPackagedContainerImplementations() {
        ClassDesc container = ClassDesc.of("com.eignex.koblas.internal.InternalMatrix");
        byte[] implementation = ClassFile.of().build(container, builder -> builder
            .withInterfaceSymbols(ClassDesc.of("com.eignex.koblas.Matrix")));
        byte[] declaration = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.host.HostBlas"), builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withMethod("window", MethodTypeDesc.of(container), ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(List.of(implementation, declaration)).size());
    }

    @Test
    void backendsRejectTypedDynamicConstantsWithGenericBootstraps() {
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        byte[] backend = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.CustomBackend"), builder -> builder
            .withInterfaceSymbols(provider)
            .withMethodBody("compute", MethodTypeDesc.of(CD_void), ClassFile.ACC_PUBLIC,
                code -> code.ldc(typedConstant(MATRIX)).pop().return_()));
        assertEquals(1, KernelBoundaries.violations(backend).size());
    }

    @Test
    void schedulesMayUsePrimitiveDynamicConstants() {
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.ldc(typedConstant(CD_double)).pop2().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void schedulesRejectContainerImplementationsWithPrimitiveConstructors() {
        ClassDesc matrix = ClassDesc.of("com.eignex.koblas.internal.InternalMatrix");
        byte[] implementation = ClassFile.of().build(matrix, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(ClassDesc.of("com.eignex.koblas.Matrix")));
        byte[] caller = fixture(DENSE_SCHEDULE, code -> code.new_(matrix).dup()
            .invokespecial(matrix, "<init>", MethodTypeDesc.of(CD_void)).pop().return_());
        assertEquals(1, KernelBoundaries.violations(List.of(implementation, caller)).size());
    }

    private static DynamicConstantDesc<?> typedConstant(ClassDesc type) {
        ClassDesc lookup = ClassDesc.of("java.lang.invoke.MethodHandles$Lookup");
        var bootstrap = MethodHandleDesc.ofMethod(STATIC, ClassDesc.of("external.Bootstrap"), "constant",
            MethodTypeDesc.of(CD_Object, lookup, CD_String, CD_Class));
        return DynamicConstantDesc.ofNamed(bootstrap, "operand", type);
    }

    @Test
    void operationPreparationMayConstructOwnedMatrices() {
        ClassDesc companion = ClassDesc.of("com.eignex.koblas.DenseMatrix$Companion");
        byte[] caller = fixture(ClassDesc.of("com.eignex.koblas.internal.operations.PreparationFixture"),
            code -> code.aconst_null()
                .invokevirtual(companion, "wrap", MethodTypeDesc.of(MATRIX)).pop().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void primitiveArithmeticCallsRemainAllowed() {
        byte[] caller = fixture(code -> code.aconst_null().iconst_0()
            .invokestatic(LEAF, "sum", MethodTypeDesc.of(CD_double, DOUBLES, CD_int)).pop2().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void primitiveDefaultBridgesRemainAllowed() {
        byte[] caller = fixture(code -> code.aconst_null().iconst_0().iconst_0().aconst_null()
            .invokestatic(LEAF, "sum$default", MethodTypeDesc.of(CD_double, DOUBLES, CD_int, CD_int, CD_Object))
            .pop2().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(strings = {"kernels/scalar/dense/ValuesKt", "scheduling/dense/BlocksKt", "operations/PreparationKt"})
    void arithmeticCallbacksMustBeSourceInlined(String path) {
        ClassDesc helper = ClassDesc.of("com.eignex.koblas.internal." + path.replace('/', '.'));
        ClassDesc function = ClassDesc.of("kotlin.jvm.functions.Function1");
        byte[] caller = fixture(code -> code.aconst_null()
            .invokestatic(helper, "visit", MethodTypeDesc.of(CD_void, function)).return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void startupAndPoolAllocationCallbacksMayRemainOutlined() {
        ClassDesc engine = ClassDesc.of("com.eignex.koblas.internal.engine.EngineSelectionKt");
        ClassDesc pool = ClassDesc.of("com.eignex.koblas.internal.storage.PooledBuffers");
        ClassDesc function = ClassDesc.of("kotlin.jvm.functions.Function1");
        byte[] caller = fixture(code -> code.aconst_null()
            .invokestatic(engine, "select", MethodTypeDesc.of(CD_Object, function)).pop()
            .aconst_null().aconst_null()
            .invokevirtual(pool, "take", MethodTypeDesc.of(CD_Object, function)).pop().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void primitiveBackendDispatchRemainsABoundary() {
        ClassDesc backend = ClassDesc.of("com.eignex.koblas.internal.kernels.EngineKernels");
        byte[] caller = fixture(code -> code.aconst_null().aconst_null().iconst_0()
            .invokeinterface(backend, "sum", MethodTypeDesc.of(CD_double, DOUBLES, CD_int)).pop2().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(strings = {"DenseVector", "DenseMatrix", "SparseVector", "SparseMatrix"})
    void backendDeclarationsRejectTypedOperandsWithoutCallSites(String name) {
        ClassDesc backend = ClassDesc.of("com.eignex.koblas.internal.host.HostBlas");
        byte[] declaration = ClassFile.of().build(backend, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withMethod("multiply", MethodTypeDesc.of(CD_void, ClassDesc.of("com.eignex.koblas." + name)),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"VectorKernels", "IndexedKernels", "PanelKernels", "PackedProductKernels", "TriangularKernels"})
    void roleImplementationsRejectTypedResultsWithoutTheirParentBytecode(String role) {
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels." + role);
        byte[] declaration = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.RoleBackend"), builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(provider)
            .withMethod("window", MethodTypeDesc.of(MATRIX), ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void backendInterfacesRejectInheritingTheTypedPublicApi() {
        ClassDesc backend = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        byte[] declaration = ClassFile.of().build(backend, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(ClassDesc.of("com.eignex.koblas.KoblasEngine")));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void backendImplementationsRejectTypedResults() {
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        byte[] declaration = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.CustomBackend"), builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(provider)
            .withMethod("window", MethodTypeDesc.of(MATRIX), ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void primitiveBackendsRejectReconstructingTypedViews() {
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        ClassDesc companion = ClassDesc.of("com.eignex.koblas.DenseMatrix$Companion");
        byte[] declaration = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.CustomBackend"), builder -> builder
            .withInterfaceSymbols(provider)
            .withMethodBody("compute", MethodTypeDesc.of(CD_void), ClassFile.ACC_PUBLIC,
                code -> code.aconst_null().invokevirtual(companion, "wrap", MethodTypeDesc.of(MATRIX)).pop().return_()));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void primitiveBackendsRejectDirectContainerConstructors() {
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        byte[] declaration = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.CustomBackend"), builder -> builder
            .withInterfaceSymbols(provider)
            .withMethodBody("compute", MethodTypeDesc.of(CD_void), ClassFile.ACC_PUBLIC,
                code -> code.new_(MATRIX).dup().invokespecial(MATRIX, "<init>", MethodTypeDesc.of(CD_void)).pop().return_()));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void primitiveBackendsRejectTypedMethodHandles() {
        ClassDesc provider = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        var factory = MethodHandleDesc.ofMethod(STATIC, LEAF, "window", MethodTypeDesc.of(MATRIX, DOUBLES));
        byte[] declaration = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.CustomBackend"), builder -> builder
            .withInterfaceSymbols(provider)
            .withMethodBody("compute", MethodTypeDesc.of(CD_void), ClassFile.ACC_PUBLIC,
                code -> code.ldc(factory).pop().return_()));
        assertEquals(1, KernelBoundaries.violations(declaration).size());
    }

    @Test
    void backendPolicyFollowsInheritedImplementationContracts() {
        ClassDesc backend = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        ClassDesc bridge = ClassDesc.of("com.eignex.koblas.internal.Bridge");
        byte[] parent = ClassFile.of().build(bridge, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(backend));
        byte[] child = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.CustomBackend"), builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(bridge)
            .withMethod("window", MethodTypeDesc.of(MATRIX), ClassFile.ACC_PUBLIC | ClassFile.ACC_ABSTRACT, method -> {}));
        assertEquals(1, KernelBoundaries.violations(List.of(parent, child)).size());
    }

    @Test
    void backendPolicyRejectsIndirectTypedApiInheritance() {
        ClassDesc backend = ClassDesc.of("com.eignex.koblas.internal.kernels.KernelProvider");
        ClassDesc bridge = ClassDesc.of("com.eignex.koblas.internal.TypedBridge");
        byte[] parent = ClassFile.of().build(bridge, builder -> builder
            .withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_INTERFACE | ClassFile.ACC_ABSTRACT)
            .withInterfaceSymbols(ClassDesc.of("com.eignex.koblas.KoblasEngine")));
        byte[] child = ClassFile.of().build(ClassDesc.of("com.eignex.koblas.internal.CustomBackend"), builder -> builder
            .withInterfaceSymbols(backend, bridge));
        assertEquals(1, KernelBoundaries.violations(List.of(parent, child)).size());
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

    @ParameterizedTest
    @ValueSource(strings = {"jdk.incubator.vector.VectorMask", "jdk.incubator.vector.VectorShuffle",
        "jdk.incubator.vector.ByteVector"})
    void vectorCarrierArraysCannotCrossProjectCalls(String name) {
        ClassDesc carrier = ClassDesc.of(name).arrayType();
        byte[] caller = fixture(code -> code.aconst_null()
            .invokestatic(VECTOR_HELPER, "consume", MethodTypeDesc.of(CD_void, carrier)).return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void cachedSpeciesMayCrossProjectCalls() {
        ClassDesc species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        byte[] caller = fixture(code -> code.invokestatic(VECTOR_HELPER, "species", MethodTypeDesc.of(species))
            .pop().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void jdkVectorIntrinsicsRemainAllowed() {
        ClassDesc species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        byte[] caller = fixture(code -> code.getstatic(VECTOR, "SPECIES_PREFERRED", species)
            .invokestatic(VECTOR, "zero", MethodTypeDesc.of(VECTOR, species)).pop().return_());
        assertTrue(KernelBoundaries.violations(caller).isEmpty());
    }

    @Test
    void methodHandlesCannotHideAVectorCall() {
        var handle = MethodHandleDesc.ofMethod(STATIC, VECTOR_HELPER, "sum", MethodTypeDesc.of(CD_double, VECTOR));
        byte[] caller = fixture(code -> code.ldc(handle).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    @Test
    void lambdaBootstrapCannotHideAVectorCall() {
        ClassDesc lookup = ClassDesc.of("java.lang.invoke.MethodHandles$Lookup");
        ClassDesc methodType = ClassDesc.of("java.lang.invoke.MethodType");
        ClassDesc methodHandle = ClassDesc.of("java.lang.invoke.MethodHandle");
        var bootstrap = MethodHandleDesc.ofMethod(STATIC, ClassDesc.of("java.lang.invoke.LambdaMetafactory"),
            "metafactory", MethodTypeDesc.of(ClassDesc.of("java.lang.invoke.CallSite"),
                lookup, CD_String, methodType, methodType, methodHandle, methodType));
        var signature = MethodTypeDesc.of(CD_double, VECTOR);
        var handle = MethodHandleDesc.ofMethod(STATIC, VECTOR_HELPER, "sum", signature);
        var dynamic = DynamicCallSiteDesc.of(bootstrap, "apply",
            MethodTypeDesc.of(ClassDesc.of("java.util.function.ToDoubleFunction")),
            MethodTypeDesc.of(CD_double, CD_Object), handle, signature);
        byte[] caller = fixture(code -> code.invokedynamic(dynamic).pop().return_());
        assertEquals(1, KernelBoundaries.violations(caller).size());
    }

    private static byte[] fixture(Consumer<CodeBuilder> body) {
        return fixture(ClassDesc.of("com.eignex.koblas.internal.KernelFixture"), body);
    }

    private static byte[] fixture(ClassDesc owner, Consumer<CodeBuilder> body) {
        return ClassFile.of().build(owner, builder ->
            builder.withMethodBody("run", MethodTypeDesc.of(CD_void),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, body));
    }
}
