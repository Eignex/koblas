package com.eignex.koblas.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.reflect.InvocationTargetException;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static java.lang.constant.ConstantDescs.*;
import static org.junit.jupiter.api.Assertions.*;

public class VectorAssertionsTest {
    private static final ClassFile CLASS_FILE = ClassFile.of();
    private static final String BACKEND = "com.eignex.koblas.internal.jvm25.simd.GuardFixture";
    private static final ClassDesc INTRINSICS = ClassDesc.of("kotlin.jvm.internal.Intrinsics");
    private static int evaluations;

    public static String message() { evaluations++; return "value"; }

    @ParameterizedTest
    @ValueSource(strings = {"checkNotNullExpressionValue", "checkExpressionValueIsNotNull", "checkNotNull"})
    void localFastPathPreservesValuesFailuresAndArgumentEvaluation(String assertion) throws Exception {
        byte[] original = fixture(BACKEND, code -> code.aload(0).dup()
            .invokestatic(ClassDesc.of(VectorAssertionsTest.class.getName()), "message", MethodTypeDesc.of(CD_String))
            .invokestatic(INTRINSICS, assertion, MethodTypeDesc.of(CD_void, CD_Object, CD_String)).areturn());
        byte[] rewritten = VectorAssertions.transform(CLASS_FILE, original);
        assertEquals(1, VectorAssertions.violations(original).size());
        assertEquals(1, KernelBoundaries.violations(original).size());
        assertTrue(VectorAssertions.violations(rewritten).isEmpty());
        assertTrue(KernelBoundaries.violations(rewritten).isEmpty());
        assertTrue(CLASS_FILE.verify(rewritten).isEmpty());
        assertArrayEquals(rewritten, VectorAssertions.transform(CLASS_FILE, rewritten));
        Object value = new Object();
        evaluations = 0;
        assertSame(value, invoke(rewritten, value));
        assertEquals(1, evaluations);
        Throwable before = failure(original, null);
        evaluations = 0;
        Throwable after = failure(rewritten, null);
        assertEquals(1, evaluations);
        assertEquals(before.getClass(), after.getClass());
        assertEquals(before.getMessage(), after.getMessage());
        assertEquals(before.getStackTrace()[0], after.getStackTrace()[0]);
    }

    @Test
    void oneArgumentCheckAndWrongTypeCastRetainTheirFailures() throws Exception {
        byte[] original = fixture(BACKEND, code -> code.aload(0).dup()
            .invokestatic(INTRINSICS, "checkNotNull", MethodTypeDesc.of(CD_void, CD_Object))
            .checkcast(CD_String).areturn());
        byte[] rewritten = VectorAssertions.transform(CLASS_FILE, original);
        assertTrue(CLASS_FILE.verify(rewritten).isEmpty());
        assertTrue(VectorAssertions.violations(rewritten).isEmpty());
        assertArrayEquals(rewritten, VectorAssertions.transform(CLASS_FILE, rewritten));
        assertEquals("value", invoke(rewritten, "value"));
        assertInstanceOf(NullPointerException.class, failure(rewritten, null));
        assertInstanceOf(ClassCastException.class, failure(rewritten, new Object()));
        assertEquals(opcodes(original, Opcode.CHECKCAST), opcodes(rewritten, Opcode.CHECKCAST));
    }

    @ParameterizedTest
    @ValueSource(strings = {"com.eignex.koblas.PublicFixture", "com.eignex.koblas.internal.kernels.scalar.GuardFixture"})
    void otherNamespacesRemainByteIdentical(String owner) {
        byte[] original = fixture(owner, code -> code.aload(0).dup().ldc("value")
            .invokestatic(INTRINSICS, "checkNotNullExpressionValue", MethodTypeDesc.of(CD_void, CD_Object, CD_String))
            .areturn());
        assertArrayEquals(original, VectorAssertions.transform(CLASS_FILE, original));
    }

    @ParameterizedTest
    @ValueSource(strings = {"checkNotNullParameter", "checkParameterIsNotNull"})
    void backendParameterChecksRemainByteIdentical(String assertion) {
        byte[] original = fixture(BACKEND, code -> code.aload(0).dup().ldc("value")
            .invokestatic(INTRINSICS, assertion, MethodTypeDesc.of(CD_void, CD_Object, CD_String)).areturn());
        assertArrayEquals(original, VectorAssertions.transform(CLASS_FILE, original));
    }

    @Test
    void wrongBranchTargetsDoNotSatisfyTheGuard() {
        byte[] original = fixture(BACKEND, code -> {
            var wrong = code.newLabel();
            var done = code.newLabel();
            code.aload(0).dup().ldc("value").dup2().pop().ifnonnull(wrong).nop()
                .labelBinding(wrong)
                .invokestatic(INTRINSICS, "checkNotNull", MethodTypeDesc.of(CD_void, CD_Object, CD_String))
                .goto_(done).pop2().labelBinding(done).areturn();
        });
        assertEquals(1, VectorAssertions.violations(original).size());
    }

    private static long opcodes(byte[] bytes, Opcode opcode) {
        return CLASS_FILE.parse(bytes).methods().getFirst().code().orElseThrow().elementStream()
            .filter(item -> item instanceof Instruction instruction && instruction.opcode() == opcode).count();
    }

    private static Throwable failure(byte[] bytes, Object argument) {
        return assertThrows(InvocationTargetException.class, () -> invoke(bytes, argument)).getCause();
    }

    private static Object invoke(byte[] bytes, Object argument) throws Exception {
        class Loader extends ClassLoader {
            Class<?> fixture() { return defineClass(null, bytes, 0, bytes.length); }
        }
        return new Loader().fixture().getMethod("run", Object.class).invoke(null, argument);
    }

    private static byte[] fixture(String owner, Consumer<CodeBuilder> body) {
        return CLASS_FILE.build(ClassDesc.of(owner), builder -> builder.withFlags(ClassFile.ACC_PUBLIC)
            .withSuperclass(CD_Object)
            .withMethodBody("run", MethodTypeDesc.of(CD_Object, CD_Object),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, body));
    }
}
