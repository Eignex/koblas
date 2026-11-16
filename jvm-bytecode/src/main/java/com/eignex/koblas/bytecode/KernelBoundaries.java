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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
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

    private static final Set<String> BACKENDS = Set.of(
        PROJECT + "internal/kernels/EngineKernels",
        PROJECT + "internal/kernels/KernelProvider",
        PROJECT + "internal/kernels/VectorKernels",
        PROJECT + "internal/kernels/IndexedKernels",
        PROJECT + "internal/kernels/PanelKernels",
        PROJECT + "internal/kernels/PackedProductKernels",
        PROJECT + "internal/kernels/TriangularKernels",
        PROJECT + "internal/host/HostBlas"
    );
    private static final Set<String> CONTAINERS = Set.of(
        PROJECT + "Vector", PROJECT + "Matrix",
        PROJECT + "DenseVector", PROJECT + "DenseMatrix",
        PROJECT + "SparseVector", PROJECT + "SparseMatrix"
    );

    private KernelBoundaries() {}

    static List<String> violations(byte[] bytes) {
        return violations(List.of(bytes));
    }

    static List<String> violations(List<byte[]> classes) {
        Map<String, Set<String>> parents = new HashMap<>();
        for (var bytes : classes) {
            var model = ClassFile.of().parse(bytes);
            var inherited = parents.computeIfAbsent(model.thisClass().asInternalName(), name -> new HashSet<>());
            model.superclass().ifPresent(parent -> inherited.add(parent.asInternalName()));
            for (var parent : model.interfaces()) inherited.add(parent.asInternalName());
        }
        var backends = descendants(parents, BACKENDS);
        var typedApi = descendants(parents, Set.of(PROJECT + "KoblasEngine"));
        var containers = descendants(parents, CONTAINERS);
        var failures = new ArrayList<String>();
        for (var bytes : classes) failures.addAll(violations(bytes, backends, typedApi, containers));
        return failures;
    }

    private static Set<String> descendants(Map<String, Set<String>> parents, Set<String> roots) {
        var result = new HashSet<>(roots);
        boolean changed;
        do {
            changed = false;
            for (var entry : parents.entrySet()) {
                if (!result.contains(entry.getKey()) && entry.getValue().stream().anyMatch(result::contains)) {
                    changed |= result.add(entry.getKey());
                }
            }
        } while (changed);
        return result;
    }

    private static List<String> violations(byte[] bytes, Set<String> backends, Set<String> typedApi, Set<String> containers) {
        var model = ClassFile.of().parse(bytes);
        String owner = model.thisClass().asInternalName();
        var failures = new ArrayList<String>(VectorAssertions.violations(bytes));
        boolean backend = backends.contains(owner);
        boolean schedule = schedule(owner);
        if (backend && typedApi.contains(owner)) {
            failures.add(owner + " inherits the typed public API; implement only primitive backend interfaces");
        }
        if (schedule && typedApi.contains(owner)) {
            failures.add(owner + " inherits the typed public API inside scheduling; pass the primitive provider once at entry");
        }
        if (schedule) {
            for (var field : model.fields()) {
                var type = field.fieldTypeSymbol();
                if (container(type, containers) || publicEngine(type, typedApi)) {
                    failures.add(owner + "." + field.fieldName().stringValue()
                        + " stores a typed operand or public engine inside scheduling; pass primitive windows once at entry");
                }
            }
        }
        for (var method : model.methods()) {
            String caller = owner + "." + method.methodName().stringValue();
            if (backend) {
                var type = method.methodTypeSymbol();
                if (typed(type, containers)) failures.add(caller + " exposes a typed container at a backend boundary; pass primitive windows");
            }
            if (schedule && publicEngine(method.methodTypeSymbol(), typedApi)) {
                failures.add(caller + " exposes the typed public engine inside scheduling; pass the primitive provider once at entry");
            }
            if (schedule && typed(method.methodTypeSymbol(), containers)) {
                failures.add(caller + " exposes a typed container inside scheduling; pass primitive windows once at entry");
            }
            if (method.code().isEmpty()) continue;
            for (var item : method.code().get()) {
                if (item instanceof InvokeInstruction call) {
                    check(caller, call.owner().asInternalName(), call.name().stringValue(),
                        call.typeSymbol(), typedApi, containers, failures);
                    if (backend && (typed(call.typeSymbol(), containers) || container(call.owner().asSymbol(), containers))) failures.add(caller + " -> "
                        + call.owner().asInternalName() + "." + call.name().stringValue()
                        + " adapts a typed container inside a backend; consume primitive windows directly");
                } else if (item instanceof InvokeDynamicInstruction dynamic) {
                    dynamicBoundary(caller, dynamic.typeSymbol(), backend, typedApi, containers, failures);
                    constant(caller, dynamic.bootstrapMethod(), backend, typedApi, containers, failures);
                    for (var argument : dynamic.bootstrapArgs()) constant(caller, argument, backend, typedApi, containers, failures);
                } else if (item instanceof LoadConstantInstruction load) {
                    constant(caller, load.constantValue(), backend, typedApi, containers, failures);
                }
            }
        }
        return failures;
    }

    private static void constant(String caller, ConstantDesc value, boolean backend, Set<String> typedApi, Set<String> containers,
            List<String> failures) {
        if (value instanceof DirectMethodHandleDesc handle && handle.lookupDescriptor().startsWith("(")) {
            String descriptor = handle.owner().descriptorString();
            var type = MethodTypeDesc.ofDescriptor(handle.lookupDescriptor());
            check(caller, descriptor.substring(1, descriptor.length() - 1), handle.methodName(), type, typedApi, containers, failures);
            if (backend && (typed(type, containers) || container(handle.owner(), containers))) failures.add(caller + " -> "
                + handle.owner().displayName() + "." + handle.methodName()
                + " adapts a typed container through a method handle inside a backend; consume primitive windows directly");
        } else if (value instanceof DynamicConstantDesc<?> dynamic) {
            dynamicBoundary(caller, MethodTypeDesc.of(dynamic.constantType()), backend, typedApi, containers, failures);
            constant(caller, dynamic.bootstrapMethod(), backend, typedApi, containers, failures);
            for (var argument : dynamic.bootstrapArgs()) constant(caller, argument, backend, typedApi, containers, failures);
        }
    }

    private static void dynamicBoundary(String caller, MethodTypeDesc type, boolean backend, Set<String> typedApi,
            Set<String> containers, List<String> failures) {
        if (backend && typed(type, containers)) {
            failures.add(caller + " adapts a typed container through a dynamic boundary inside a backend; consume primitive windows directly");
        }
        if (schedule(caller) && (typed(type, containers) || publicEngine(type, typedApi))) {
            failures.add(caller + " carries a typed operand or public engine through a dynamic boundary inside scheduling; pass primitive windows once at entry");
        }
    }

    private static void check(String caller, String owner, String name, MethodTypeDesc type,
            Set<String> typedApi, Set<String> containers, List<String> failures) {
        if (schedule(caller)) {
            if (typed(type, containers) || containers.contains(owner)) {
                failures.add(caller + " -> " + owner + "." + name
                    + " carries a typed container inside scheduling; pass arrays, offsets, strides and extents");
            }
            if (typedApi.contains(owner) || publicEngine(type, typedApi)) {
                failures.add(caller + " -> " + owner + "." + name
                    + " reaches the typed public engine inside scheduling; pass the primitive provider once at entry");
            }
        }
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

    private static boolean typed(MethodTypeDesc type, Set<String> containers) {
        if (container(type.returnType(), containers)) return true;
        for (var parameter : type.parameterArray()) if (container(parameter, containers)) return true;
        return false;
    }

    private static boolean schedule(String owner) {
        return owner.startsWith(PROJECT + "internal/scheduling/");
    }

    private static boolean publicEngine(MethodTypeDesc type, Set<String> typedApi) {
        if (publicEngine(type.returnType(), typedApi)) return true;
        for (var parameter : type.parameterArray()) if (publicEngine(parameter, typedApi)) return true;
        return false;
    }

    private static boolean publicEngine(ClassDesc type, Set<String> typedApi) {
        while (type.isArray()) type = type.componentType();
        String descriptor = type.descriptorString();
        return descriptor.startsWith("L") && typedApi.contains(descriptor.substring(1, descriptor.length() - 1));
    }

    private static boolean container(ClassDesc type, Set<String> containers) {
        while (type.isArray()) type = type.componentType();
        String descriptor = type.descriptorString();
        return descriptor.startsWith("L") && containers.contains(descriptor.substring(1, descriptor.length() - 1));
    }

    private static boolean vector(ClassDesc type) {
        while (type.isArray()) type = type.componentType();
        return CARRIERS.contains(type.descriptorString());
    }

    public static void main(String[] arguments) throws Exception {
        if (arguments.length != 1) throw new IllegalArgumentException("Expected the JVM JAR path");
        var classes = new ArrayList<byte[]>();
        try (var jar = new ZipFile(Path.of(arguments[0]).toFile())) {
            var entries = jar.entries();
            while (entries.hasMoreElements()) {
                var entry = entries.nextElement();
                if (!entry.getName().endsWith(".class") || entry.getName().endsWith("module-info.class")) continue;
                try (var stream = jar.getInputStream(entry)) { classes.add(stream.readAllBytes()); }
            }
        }
        var failures = violations(classes);
        if (!failures.isEmpty()) throw new IllegalStateException(String.join("\n", failures));
        System.out.println("Verified arithmetic call boundaries in " + classes.size() + " packaged classes");
    }
}
