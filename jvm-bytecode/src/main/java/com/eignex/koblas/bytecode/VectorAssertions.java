package com.eignex.koblas.bytecode;

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassHierarchyResolver;
import java.lang.classfile.CodeModel;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.MethodTransform;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/** Keeps Vector backend null failures while making the successful path local to its kernel. */
public final class VectorAssertions {
    private static final String BACKEND = "com/eignex/koblas/internal/jvm25/simd/";
    private static final Set<String> ASSERTIONS = Set.of(
        "checkNotNullExpressionValue", "checkExpressionValueIsNotNull", "checkNotNull"
    );

    private VectorAssertions() {}

    private static int arguments(Instruction instruction) {
        if (!(instruction instanceof InvokeInstruction call) || call.opcode() != Opcode.INVOKESTATIC
                || !call.owner().asInternalName().equals("kotlin/jvm/internal/Intrinsics")
                || !ASSERTIONS.contains(call.name().stringValue())) return 0;
        return switch (call.type().stringValue()) {
            case "(Ljava/lang/Object;)V" -> 1;
            case "(Ljava/lang/Object;Ljava/lang/String;)V" -> 2;
            default -> 0;
        };
    }

    private static Set<Integer> pending(CodeModel code) {
        var instructions = new ArrayList<Instruction>();
        var labels = new HashMap<Label, Integer>();
        for (var element : code) {
            if (element instanceof LabelTarget target) labels.put(target.label(), instructions.size());
            if (element instanceof Instruction instruction) instructions.add(instruction);
        }
        var result = new HashSet<Integer>();
        for (int i = 0; i < instructions.size(); i++) {
            int count = arguments(instructions.get(i));
            if (count > 0 && !guarded(instructions, labels, i, count)) result.add(i);
        }
        return result;
    }

    private static boolean guarded(List<Instruction> code, Map<Label, Integer> labels, int i, int count) {
        int start = i - count - 1;
        if (start < 0 || i + 2 >= code.size()) return false;
        if (code.get(start).opcode() != (count == 2 ? Opcode.DUP2 : Opcode.DUP)) return false;
        if (count == 2 && code.get(start + 1).opcode() != Opcode.POP) return false;
        return code.get(i - 1) instanceof BranchInstruction test && test.opcode() == Opcode.IFNONNULL
            && Integer.valueOf(i + 2).equals(labels.get(test.target()))
            && code.get(i + 1) instanceof BranchInstruction exit && exit.opcode() == Opcode.GOTO
            && Integer.valueOf(i + 3).equals(labels.get(exit.target()))
            && code.get(i + 2).opcode() == (count == 2 ? Opcode.POP2 : Opcode.POP);
    }

    static List<String> violations(byte[] bytes) {
        var model = ClassFile.of().parse(bytes);
        var failures = new ArrayList<String>();
        String owner = model.thisClass().asInternalName();
        if (!owner.startsWith(BACKEND)) return failures;
        for (var method : model.methods()) {
            if (method.code().isEmpty()) continue;
            for (int index : pending(method.code().get())) failures.add(owner + "."
                + key(method) + " has an opaque null assertion at instruction " + index);
        }
        return failures;
    }

    static byte[] transform(ClassFile classFile, byte[] bytes) {
        var model = classFile.parse(bytes);
        if (!model.thisClass().asInternalName().startsWith(BACKEND)) return bytes;
        var methods = new HashMap<String, Set<Integer>>();
        for (var method : model.methods()) {
            if (method.code().isEmpty()) continue;
            var positions = pending(method.code().get());
            if (!positions.isEmpty()) methods.put(key(method), positions);
        }
        if (methods.isEmpty()) return bytes;
        return classFile.transformClass(model, (builder, element) -> {
            if (element instanceof MethodModel method && methods.containsKey(key(method))) {
                Set<Integer> positions = methods.get(key(method));
                builder.transformMethod(method, MethodTransform.transformingCode(CodeTransform.ofStateful(() -> {
                    int[] index = {0};
                    return (code, item) -> {
                        if (item instanceof Instruction instruction && positions.contains(index[0]++)) {
                            var nonNull = code.newLabel();
                            var done = code.newLabel();
                            // Arguments have already been evaluated. Only null reaches the original failure helper.
                            if (arguments(instruction) == 2) code.dup2().pop(); else code.dup();
                            code.ifnonnull(nonNull).with(item).goto_(done).labelBinding(nonNull);
                            if (arguments(instruction) == 2) code.pop2(); else code.pop();
                            code.labelBinding(done);
                        } else code.with(item);
                    };
                })));
            } else builder.with(element);
        });
    }

    private static String key(MethodModel method) {
        return method.methodName().stringValue() + method.methodType().stringValue();
    }

    public static void main(String[] arguments) throws Exception {
        if (arguments.length != 1) throw new IllegalArgumentException("Expected the JVM JAR path");
        Path jar = Path.of(arguments[0]);
        Path temporary = Files.createTempFile(jar.toAbsolutePath().getParent(), "vector-assertions-", ".jar");
        try {
            try (var loader = new URLClassLoader(new URL[] {jar.toUri().toURL()}, VectorAssertions.class.getClassLoader())) {
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
        try (var input = new ZipFile(jar.toFile()); var stream = new ZipOutputStream(Files.newOutputStream(output))) {
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
        System.out.println("Localized Vector assertions in " + changed + " classes");
    }
}
