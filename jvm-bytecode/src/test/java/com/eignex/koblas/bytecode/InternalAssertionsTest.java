package com.eignex.koblas.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import jdk.incubator.vector.DoubleVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static java.lang.constant.ConstantDescs.*;
import static org.junit.jupiter.api.Assertions.*;

public class InternalAssertionsTest {
    private static final ClassFile CLASS_FILE = ClassFile.of();
    private static final ClassDesc INTRINSICS = ClassDesc.of("kotlin.jvm.internal.Intrinsics");
    private static final MethodTypeDesc ASSERTION = MethodTypeDesc.of(CD_void, CD_Object, CD_String);
    private static final String INTERNAL = "com.eignex.koblas.internal.PolicyFixture";
    private static int evaluations;

    public static Object evaluate() { evaluations++; return null; }

    @ParameterizedTest
    @ValueSource(strings = {"checkNotNullParameter", "checkParameterIsNotNull",
        "checkNotNullExpressionValue", "checkExpressionValueIsNotNull"})
    void publicAssertionsRemainUnchanged(String name) {
        byte[] original = assertionFixture("com.eignex.koblas.PublicFixture", name);
        assertArrayEquals(original, InternalAssertions.transform(CLASS_FILE, original));
    }

    @ParameterizedTest
    @ValueSource(strings = {"checkNotNullParameter", "checkParameterIsNotNull",
        "checkNotNullExpressionValue", "checkExpressionValueIsNotNull"})
    void generatedInternalAssertionsAreRemovedWithoutDroppingArgumentEvaluation(String name) throws Exception {
        byte[] original = fixture(INTERNAL, code -> code
            .invokestatic(ClassDesc.of(InternalAssertionsTest.class.getName()), "evaluate",
                MethodTypeDesc.of(CD_Object))
            .dup().ldc("value").invokestatic(INTRINSICS, name, ASSERTION).areturn());
        byte[] rewritten = InternalAssertions.transform(CLASS_FILE, original);
        evaluations = 0;
        assertNull(invoke(rewritten, null));
        assertEquals(1, evaluations);
        assertEquals(0, intrinsicCalls(rewritten));
    }

    @Test
    void explicitNullValidationStillThrows() {
        byte[] rewritten = InternalAssertions.transform(CLASS_FILE, assertionFixture(INTERNAL, "checkNotNull"));
        var failure = assertThrows(InvocationTargetException.class, () -> invoke(rewritten, null));
        assertInstanceOf(NullPointerException.class, failure.getCause());
        assertEquals("value", failure.getCause().getMessage());
        assertEquals(1, intrinsicCalls(rewritten));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void vectorProducerLosesOnlyItsRedundantNullAssertion(boolean spilled) throws Exception {
        ClassDesc doubles = ClassDesc.of("jdk.incubator.vector.DoubleVector");
        ClassDesc longs = ClassDesc.of("jdk.incubator.vector.LongVector");
        ClassDesc ints = ClassDesc.of("jdk.incubator.vector.IntVector");
        ClassDesc species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        byte[] original = fixture(INTERNAL, code -> {
            code.aload(0).checkcast(doubles).getstatic(longs, "SPECIES_PREFERRED", species).iconst_0()
                .invokevirtual(doubles, "reinterpretShape", MethodTypeDesc.of(
                    ClassDesc.of("jdk.incubator.vector.Vector"), species, CD_int));
            if (spilled) code.astore(1).aload(1);
            else code.dup();
            code.ldc("null cannot be cast to non-null type jdk.incubator.vector.IntVector")
                .invokestatic(INTRINSICS, "checkNotNull", ASSERTION);
            if (spilled) code.aload(1);
            code.checkcast(ints).areturn();
        });
        byte[] rewritten = InternalAssertions.transform(CLASS_FILE, original);
        assertEquals(0, intrinsicCalls(rewritten));
        assertEquals(opcodes(original, Opcode.CHECKCAST), opcodes(rewritten, Opcode.CHECKCAST));
        var failure = assertThrows(InvocationTargetException.class,
            () -> invoke(rewritten, DoubleVector.zero(DoubleVector.SPECIES_PREFERRED)));
        assertInstanceOf(ClassCastException.class, failure.getCause());
    }

    @Test
    void alternateEntryRetainsTheCastNullAssertion() {
        ClassDesc doubles = ClassDesc.of("jdk.incubator.vector.DoubleVector");
        ClassDesc longs = ClassDesc.of("jdk.incubator.vector.LongVector");
        ClassDesc species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        byte[] original = fixture(INTERNAL, code -> {
            var alternate = code.newLabel();
            code.aload(0).dup().ifnull(alternate).pop()
                .aload(0).checkcast(doubles).getstatic(longs, "SPECIES_PREFERRED", species).iconst_0()
                .invokevirtual(doubles, "reinterpretShape", MethodTypeDesc.of(
                    ClassDesc.of("jdk.incubator.vector.Vector"), species, CD_int))
                .labelBinding(alternate).dup()
                .ldc("null cannot be cast to non-null type jdk.incubator.vector.LongVector")
                .invokestatic(INTRINSICS, "checkNotNull", ASSERTION).checkcast(longs).areturn();
        });
        byte[] rewritten = InternalAssertions.transform(CLASS_FILE, original);
        assertEquals(1, intrinsicCalls(rewritten));
        var failure = assertThrows(InvocationTargetException.class, () -> invoke(rewritten, null));
        assertInstanceOf(NullPointerException.class, failure.getCause());
    }

    @Test
    void unknownCastProducerRetainsItsNullAssertion() {
        byte[] original = fixture(INTERNAL, code -> code.aload(0).dup()
            .ldc("null cannot be cast to non-null type jdk.incubator.vector.LongVector")
            .invokestatic(INTRINSICS, "checkNotNull", ASSERTION)
            .checkcast(ClassDesc.of("jdk.incubator.vector.LongVector")).areturn());
        byte[] rewritten = InternalAssertions.transform(CLASS_FILE, original);
        assertEquals(1, intrinsicCalls(rewritten));
        var failure = assertThrows(InvocationTargetException.class, () -> invoke(rewritten, null));
        assertInstanceOf(NullPointerException.class, failure.getCause());
    }

    @Test
    void packagedMetadataAndPublicClassesArePreserved(@TempDir Path directory) throws Exception {
        Path jar = directory.resolve("fixture.jar");
        byte[] metadata = {1, 2, 3, 4};
        byte[] publicClass = assertionFixture("com.eignex.koblas.PublicFixture", "checkNotNullParameter");
        try (var output = new ZipOutputStream(Files.newOutputStream(jar))) {
            output.putNextEntry(new ZipEntry("META-INF/main.kotlin_module"));
            output.write(metadata);
            output.closeEntry();
            output.putNextEntry(new ZipEntry("com/eignex/koblas/PublicFixture.class"));
            output.write(publicClass);
            output.closeEntry();
        }
        InternalAssertions.main(new String[] {jar.toString()});
        try (var input = new ZipFile(jar.toFile())) {
            assertArrayEquals(metadata, input.getInputStream(input.getEntry("META-INF/main.kotlin_module")).readAllBytes());
            assertArrayEquals(publicClass, input.getInputStream(input.getEntry("com/eignex/koblas/PublicFixture.class")).readAllBytes());
        }
    }

    private static byte[] assertionFixture(String name, String assertion) {
        return fixture(name, code -> code.aload(0).dup().ldc("value")
            .invokestatic(INTRINSICS, assertion, ASSERTION).areturn());
    }

    private static byte[] fixture(String name, Consumer<CodeBuilder> body) {
        return CLASS_FILE.build(ClassDesc.of(name), builder -> builder.withFlags(ClassFile.ACC_PUBLIC)
            .withSuperclass(CD_Object)
            .withMethodBody("run", MethodTypeDesc.of(CD_Object, CD_Object),
                ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, body));
    }

    private static Object invoke(byte[] bytes, Object argument) throws Exception {
        class Loader extends ClassLoader {
            Class<?> fixture() { return defineClass(null, bytes, 0, bytes.length); }
        }
        return new Loader().fixture().getMethod("run", Object.class).invoke(null, argument);
    }

    private static long intrinsicCalls(byte[] bytes) {
        return CLASS_FILE.parse(bytes).methods().stream().flatMap(method -> method.code().stream())
            .flatMap(code -> code.elementStream())
            .filter(item -> item instanceof InvokeInstruction call
                && call.owner().asInternalName().equals("kotlin/jvm/internal/Intrinsics")).count();
    }

    private static long opcodes(byte[] bytes, Opcode opcode) {
        return CLASS_FILE.parse(bytes).methods().stream().flatMap(method -> method.code().stream())
            .flatMap(code -> code.elementStream())
            .filter(item -> item instanceof Instruction instruction && instruction.opcode() == opcode).count();
    }
}
