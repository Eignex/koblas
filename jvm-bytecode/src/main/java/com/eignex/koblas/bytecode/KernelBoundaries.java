package com.eignex.koblas.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.instruction.InvokeDynamicInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.ConstantInstruction.LoadConstantInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDesc;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.DynamicConstantDesc;
import java.lang.constant.MethodTypeDesc;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.zip.ZipFile;

/** Checks the compiled arithmetic boundaries, independently of Kotlin metadata and JIT heuristics. */
public final class KernelBoundaries {
    private static final String PROJECT = "com/eignex/koblas/";
    private static final Set<String> CARRIERS = Set.of(
        "Ljdk/incubator/vector/Vector;", "Ljdk/incubator/vector/DoubleVector;",
        "Ljdk/incubator/vector/LongVector;", "Ljdk/incubator/vector/IntVector;",
        "Ljdk/incubator/vector/FloatVector;", "Ljdk/incubator/vector/ShortVector;",
        "Ljdk/incubator/vector/ByteVector;", "Ljdk/incubator/vector/VectorMask;",
        "Ljdk/incubator/vector/VectorShuffle;"
    );

    private KernelBoundaries() {}

    static List<String> violations(byte[] bytes) {
        var model = ClassFile.of().parse(bytes);
        String owner = model.thisClass().asInternalName();
        var failures = new ArrayList<String>(VectorAssertions.violations(bytes));
        for (var method : model.methods()) {
            if (method.code().isEmpty()) continue;
            String caller = owner + "." + method.methodName().stringValue();
            for (var item : method.code().get()) {
                if (item instanceof InvokeInstruction call) {
                    check(caller, call.owner().asInternalName(), call.name().stringValue(),
                        call.typeSymbol(), failures);
                } else if (item instanceof InvokeDynamicInstruction dynamic) {
                    constant(caller, dynamic.bootstrapMethod(), failures);
                    for (var argument : dynamic.bootstrapArgs()) constant(caller, argument, failures);
                } else if (item instanceof LoadConstantInstruction load) {
                    constant(caller, load.constantValue(), failures);
                }
            }
        }
        return failures;
    }

    private static void constant(String caller, ConstantDesc value, List<String> failures) {
        if (value instanceof DirectMethodHandleDesc handle && handle.lookupDescriptor().startsWith("(")) {
            String descriptor = handle.owner().descriptorString();
            check(caller, descriptor.substring(1, descriptor.length() - 1), handle.methodName(),
                MethodTypeDesc.ofDescriptor(handle.lookupDescriptor()), failures);
        } else if (value instanceof DynamicConstantDesc<?> dynamic) {
            constant(caller, dynamic.bootstrapMethod(), failures);
            for (var argument : dynamic.bootstrapArgs()) constant(caller, argument, failures);
        }
    }

    private static void check(String caller, String owner, String name, MethodTypeDesc type,
            List<String> failures) {
        if (!owner.startsWith(PROJECT)) return;
        boolean carrier = vector(type.returnType());
        for (var parameter : type.parameterArray()) carrier |= vector(parameter);
        boolean arithmetic = owner.startsWith(PROJECT + "internal/kernels/")
            || owner.startsWith(PROJECT + "internal/scheduling/")
            || owner.startsWith(PROJECT + "internal/operations/");
        boolean genericCallback = false;
        for (var parameter : type.parameterArray()) {
            genericCallback |= parameter.descriptorString().startsWith("Lkotlin/jvm/functions/Function");
        }
        if (arithmetic && genericCallback) failures.add(caller + " -> " + owner + "." + name + type.descriptorString()
            + " carries an erased callback into arithmetic; source-inline the helper or use a primitive callback signature");
        if (carrier) failures.add(caller + " -> " + owner + "." + name + type.descriptorString()
            + " carries a Vector API value across a project call; source-inline the helper into its concrete caller");
    }

    private static boolean vector(ClassDesc type) {
        while (type.isArray()) type = type.componentType();
        return CARRIERS.contains(type.descriptorString());
    }

    public static void main(String[] arguments) throws Exception {
        if (arguments.length != 1) throw new IllegalArgumentException("Expected the JVM JAR path");
        var failures = new ArrayList<String>();
        int classes = 0;
        try (var jar = new ZipFile(Path.of(arguments[0]).toFile())) {
            var entries = jar.entries();
            while (entries.hasMoreElements()) {
                var entry = entries.nextElement();
                if (!entry.getName().endsWith(".class") || entry.getName().endsWith("module-info.class")) continue;
                try (var stream = jar.getInputStream(entry)) { failures.addAll(violations(stream.readAllBytes())); }
                classes++;
            }
        }
        if (!failures.isEmpty()) throw new IllegalStateException(String.join("\n", failures));
        System.out.println("Verified arithmetic call boundaries in " + classes + " packaged classes");
    }
}
