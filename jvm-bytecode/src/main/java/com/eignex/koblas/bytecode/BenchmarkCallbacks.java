package com.eignex.koblas.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.instruction.InvokeDynamicInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipFile;

/** Checks primitive callback signatures and their benchmark-side call paths before measuring kernels. */
public final class BenchmarkCallbacks {
    private static final String BENCH = "com/eignex/koblas/bench/";
    private static final String OPERATION = BENCH + "BenchmarkOperation";
    private static final String OPERATION_TYPE = "L" + OPERATION + ";";
    private static final Set<String> BOXES = Set.of(
        "java/lang/Double", "java/lang/Float", "java/lang/Long", "java/lang/Integer",
        "java/lang/Short", "java/lang/Byte", "java/lang/Character", "java/lang/Boolean"
    );
    private record Method(String owner, String name, String type) {}

    private BenchmarkCallbacks() {}

    static List<String> violations(Map<String, byte[]> classes) {
        var models = new HashMap<String, ClassModel>();
        var methods = new HashMap<Method, MethodModel>();
        var roots = new HashSet<Method>();
        var failures = new ArrayList<String>();
        for (var bytes : classes.values()) {
            var model = ClassFile.of().parse(bytes);
            String owner = model.thisClass().asInternalName();
            if (owner.startsWith(BENCH)) models.put(owner, model);
        }
        Set<String> operationTypes = subtypes(models, OPERATION);
        for (String type : operationTypes) roots.add(new Method(type, "invoke", "()D"));
        for (var model : models.values()) {
            String owner = model.thisClass().asInternalName();
            for (var method : model.methods()) {
                var key = new Method(owner, method.methodName().stringValue(), method.methodType().stringValue());
                methods.put(key, method);
                if (method.code().isEmpty()) continue;
                for (var item : method.code().get()) {
                    if (!(item instanceof InvokeDynamicInstruction factory)
                            || !operationTypes.contains(factory.typeSymbol().returnType().descriptorString()
                                .replaceFirst("^L", "").replaceFirst(";$", ""))) continue;
                    for (var argument : factory.bootstrapArgs()) {
                        if (!(argument instanceof DirectMethodHandleDesc target)
                                || !target.lookupDescriptor().startsWith("(")) continue;
                        var type = MethodTypeDesc.ofDescriptor(target.lookupDescriptor());
                        if (!type.returnType().descriptorString().equals("D")) {
                            failures.add(key + " adapts a boxed callback result from " + target);
                        }
                        String descriptor = target.owner().descriptorString();
                        roots.add(new Method(descriptor.substring(1, descriptor.length() - 1),
                            target.methodName(), target.lookupDescriptor()));
                    }
                }
            }
        }
        var operation = models.get(OPERATION);
        if (operation == null || !methods.containsKey(new Method(OPERATION, "invoke", "()D"))
                || operation.interfaces().stream().anyMatch(type ->
                    type.asInternalName().startsWith("kotlin/jvm/functions/Function"))) {
            failures.add("BenchmarkOperation must expose a primitive invoke()D without a generic Function bridge");
        }
        if (!methods.containsKey(new Method(BENCH + "CaseWork", "getRun", "()" + OPERATION_TYPE))) {
            failures.add("CaseWork.run must store BenchmarkOperation");
        }
        roots.add(new Method(BENCH + "JvmCaseWork", "run", "()D"));
        var pending = new ArrayDeque<>(roots);
        var visited = new HashSet<Method>();
        var dispatch = new HashMap<String, Set<String>>();
        while (!pending.isEmpty()) {
            var key = pending.removeFirst();
            if (!visited.add(key)) continue;
            var method = methods.get(key);
            if (method == null) {
                // A receiver can implement an interface with a body inherited from a non-implementing superclass.
                var model = models.get(key.owner());
                if (model != null) {
                    model.superclass().ifPresent(parent ->
                        pending.addLast(new Method(parent.asInternalName(), key.name(), key.type())));
                    for (var parent : model.interfaces()) {
                        pending.addLast(new Method(parent.asInternalName(), key.name(), key.type()));
                    }
                }
                continue;
            }
            if (method.code().isEmpty()) continue;
            for (var item : method.code().get()) {
                if (!(item instanceof InvokeInstruction call)) continue;
                String owner = call.owner().asInternalName();
                String name = call.name().stringValue();
                if (owner.startsWith("kotlin/jvm/functions/Function") && name.equals("invoke")
                        || BOXES.contains(owner) && name.equals("valueOf")
                        || (BOXES.contains(owner) || owner.equals("java/lang/Number"))
                            && name.endsWith("Value") && call.typeSymbol().parameterCount() == 0) {
                    failures.add(key + " crosses a boxed callback boundary through " + owner + "." + name);
                }
                // Only the harness is checked here. Allocations inside the measured operation remain observable.
                if (owner.startsWith(BENCH)) {
                    Set<String> targets = call.opcode() == java.lang.classfile.Opcode.INVOKEINTERFACE
                            || call.opcode() == java.lang.classfile.Opcode.INVOKEVIRTUAL
                        ? dispatch.computeIfAbsent(owner, type -> subtypes(models, type)) : Set.of(owner);
                    for (String target : targets) pending.addLast(new Method(target, name, call.type().stringValue()));
                }
            }
        }
        return failures;
    }

    private static Set<String> subtypes(Map<String, ClassModel> models, String parent) {
        var types = new HashSet<String>();
        types.add(parent);
        boolean changed;
        do {
            changed = false;
            for (var model : models.values()) {
                if (model.superclass().map(type -> types.contains(type.asInternalName())).orElse(false)
                        || model.interfaces().stream().anyMatch(type -> types.contains(type.asInternalName()))) {
                    changed |= types.add(model.thisClass().asInternalName());
                }
            }
        } while (changed);
        return types;
    }

    public static void main(String[] arguments) throws Exception {
        if (arguments.length != 1) throw new IllegalArgumentException("Expected the benchmark JVM JAR path");
        var classes = new HashMap<String, byte[]>();
        try (var jar = new ZipFile(Path.of(arguments[0]).toFile())) {
            var entries = jar.entries();
            while (entries.hasMoreElements()) {
                var entry = entries.nextElement();
                if (!entry.getName().startsWith(BENCH) || !entry.getName().endsWith(".class")) continue;
                try (var source = jar.getInputStream(entry)) { classes.put(entry.getName(), source.readAllBytes()); }
            }
        }
        var failures = violations(classes);
        if (!failures.isEmpty()) throw new IllegalStateException(String.join("\n", failures));
        System.out.println("Verified primitive benchmark callbacks in " + classes.size() + " classes");
    }
}
