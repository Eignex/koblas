package com.eignex.koblas.bytecode;

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassHierarchyResolver;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.Instruction;
import java.lang.classfile.MethodModel;
import java.lang.classfile.MethodTransform;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.ConstantInstruction.LoadConstantInstruction;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.classfile.instruction.TypeCheckInstruction;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/** Build-time policy for trusted implementation code; never shipped as a runtime dependency. */
public final class InternalAssertions {
    private static final String INTERNAL = "com/eignex/koblas/internal/";
    private static final String INTRINSICS = "kotlin/jvm/internal/Intrinsics";
    private static final String ASSERTION_TYPE = "(Ljava/lang/Object;Ljava/lang/String;)V";
    private static final Set<String> GENERATED = Set.of(
        "checkNotNullParameter", "checkParameterIsNotNull",
        "checkNotNullExpressionValue", "checkExpressionValueIsNotNull"
    );
    private static final Set<String> VECTORS = Set.of(
        "jdk/incubator/vector/DoubleVector", "jdk/incubator/vector/LongVector",
        "jdk/incubator/vector/IntVector"
    );

    private InternalAssertions() {}

    static byte[] transform(ClassFile classFile, byte[] bytes) {
        var model = classFile.parse(bytes);
        if (!model.thisClass().asInternalName().startsWith(INTERNAL)) return bytes;
        var removals = new java.util.HashMap<String, Set<Integer>>();
        for (var method : model.methods()) {
            if (method.code().isEmpty()) continue;
            Set<Integer> positions = vectorCastAssertions(method);
            int index = 0;
            for (var item : method.code().get()) {
                if (item instanceof Instruction) {
                    if (item instanceof InvokeInstruction call && assertion(call)
                            && GENERATED.contains(call.name().stringValue())) positions.add(index);
                    index++;
                }
            }
            if (!positions.isEmpty()) removals.put(methodKey(method), positions);
        }
        if (removals.isEmpty()) return bytes;
        return classFile.transformClass(model, (builder, element) -> {
            if (element instanceof MethodModel method && removals.containsKey(methodKey(method))) {
                Set<Integer> positions = removals.get(methodKey(method));
                builder.transformMethod(method, MethodTransform.transformingCode(CodeTransform.ofStateful(() -> {
                    int[] index = {0};
                    return (code, item) -> {
                        if (item instanceof Instruction) {
                            int position = index[0]++;
                            if (positions.contains(position)) {
                                // Keep argument evaluation, including side effects; discard only the assertion.
                                code.pop2();
                                return;
                            }
                        }
                        code.with(item);
                    };
                })));
            } else builder.with(element);
        });
    }

    private static String methodKey(MethodModel method) {
        return method.methodName().stringValue() + method.methodType().stringValue();
    }

    private static boolean assertion(InvokeInstruction call) {
        return call.opcode() == Opcode.INVOKESTATIC
            && call.owner().asInternalName().equals(INTRINSICS)
            && call.type().stringValue().equals(ASSERTION_TYPE);
    }

    /** Removes only cast assertions immediately fed by the non-null Vector API reinterpretation. */
    private static Set<Integer> vectorCastAssertions(MethodModel method) {
        var removable = new java.util.HashSet<Integer>();
        var instructions = new ArrayList<Instruction>();
        var labels = new java.util.HashSet<Integer>();
        for (var element : method.code().get()) {
            if (element instanceof LabelTarget) labels.add(instructions.size());
            if (element instanceof Instruction instruction) instructions.add(instruction);
        }
        for (int i = 3; i + 1 < instructions.size(); i++) {
            if (!(instructions.get(i) instanceof InvokeInstruction call) || !assertion(call)
                    || !call.name().stringValue().equals("checkNotNull")
                    || !(instructions.get(i - 1) instanceof LoadConstantInstruction message)
                    || !(message.constantValue() instanceof String text)
                    || !text.startsWith("null cannot be cast to non-null type jdk.incubator.vector.")) continue;
            int producer;
            if (instructions.get(i - 2).opcode() == Opcode.DUP && vectorCast(instructions.get(i + 1))) {
                producer = i - 3;
            } else if (i >= 4 && i + 2 < instructions.size()
                    && instructions.get(i - 3) instanceof StoreInstruction store
                    && instructions.get(i - 2) instanceof LoadInstruction load
                    && instructions.get(i + 1) instanceof LoadInstruction result
                    && store.slot() == load.slot() && store.slot() == result.slot()
                    && vectorCast(instructions.get(i + 2))) {
                producer = i - 4;
            } else continue;
            if (!reinterpretation(instructions.get(producer))) continue;
            // Reject alternate entries into the producer/assertion sequence.
            boolean alternateEntry = false;
            for (int j = producer + 1; j <= i; j++) alternateEntry |= labels.contains(j);
            if (!alternateEntry) removable.add(i);
        }
        return removable;
    }

    private static boolean reinterpretation(Instruction instruction) {
        return instruction instanceof InvokeInstruction producer
            && producer.opcode() == Opcode.INVOKEVIRTUAL
            && VECTORS.contains(producer.owner().asInternalName())
            && producer.name().stringValue().equals("reinterpretShape")
            && producer.type().stringValue().equals(
                "(Ljdk/incubator/vector/VectorSpecies;I)Ljdk/incubator/vector/Vector;");
    }

    private static boolean vectorCast(Instruction instruction) {
        return instruction instanceof TypeCheckInstruction cast
            && cast.opcode() == Opcode.CHECKCAST && VECTORS.contains(cast.type().asInternalName());
    }

    public static void main(String[] arguments) throws Exception {
        if (arguments.length != 1) throw new IllegalArgumentException("Expected the JVM JAR path");
        Path jar = Path.of(arguments[0]);
        var temporary = Files.createTempFile(jar.toAbsolutePath().getParent(), "internal-assertions-", ".jar");
        try {
            try (var loader = new URLClassLoader(new URL[] {jar.toUri().toURL()}, InternalAssertions.class.getClassLoader())) {
                var classFile = ClassFile.of(ClassFile.ClassHierarchyResolverOption.of(
                    ClassHierarchyResolver.ofResourceParsing(loader).cached()));
                rewrite(jar, temporary, classFile);
            }
            Files.move(temporary, jar, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static void rewrite(Path jar, Path output, ClassFile classFile) throws IOException {
        int changed = 0;
        try (var input = new ZipFile(jar.toFile());
                var stream = new ZipOutputStream(Files.newOutputStream(output))) {
            var entries = input.entries();
            while (entries.hasMoreElements()) {
                var entry = entries.nextElement();
                byte[] bytes;
                try (var source = input.getInputStream(entry)) { bytes = source.readAllBytes(); }
                if (entry.getName().endsWith(".class") && !entry.getName().endsWith("module-info.class")) {
                    byte[] rewritten = transform(classFile, bytes);
                    var errors = classFile.verify(rewritten);
                    if (!errors.isEmpty()) throw new IOException("Invalid bytecode in " + entry.getName() + ": " + errors);
                    if (!java.util.Arrays.equals(bytes, rewritten)) changed++;
                    bytes = rewritten;
                }
                var copy = new ZipEntry(entry.getName());
                copy.setTime(entry.getTime());
                stream.putNextEntry(copy);
                stream.write(bytes);
                stream.closeEntry();
            }
        }
        System.out.println("Internal assertion policy processed " + changed + " classes");
    }
}
