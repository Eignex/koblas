import java.lang.foreign.*;
import java.lang.invoke.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import static java.lang.foreign.ValueLayout.*;

/** Diagnostic only: compare JVM reductions with native buffers at matched and fixed 64-byte alignments. */
public class KotlinAlignmentComparison {
    static final int N = 65536;
    static final int ROWS = 256;
    static final int ITERATIONS = Integer.getInteger("trial.iterations", 1000);
    static final int SAMPLES = Integer.getInteger("trial.samples", 5);
    static final int ROUNDS = Integer.getInteger("trial.rounds", 3);
    static final Linker LINKER = Linker.nativeLinker();
    static MethodHandle alignment;
    static volatile double sink;
    record Policy(String name, MethodHandle known, MethodHandle prefix) {}

    static MethodHandle function(SymbolLookup symbols, String name, FunctionDescriptor descriptor) {
        return LINKER.downcallHandle(symbols.find(name).orElseThrow(), descriptor);
    }

    static long pageOffset(MemorySegment array) throws Throwable {
        // memchr returns its input pointer when the first byte matches; never dereference the returned address.
        int first = Byte.toUnsignedInt(array.get(JAVA_BYTE, 0));
        MemorySegment address = (MemorySegment) alignment.invokeExact(array, first, 1L);
        if (address.address() == 0) throw new AssertionError("address snapshot failed");
        return address.address() & 4095;
    }

    static final class Fixture {
        final double[] a, b, y;
        final int offset;
        final MemorySegment heapA, heapB, heapY, rootA, rootB, rootY, alignedA, alignedB;
        MemorySegment nativeA, nativeB, nativeY;

        Fixture(double[] a, double[] b, int offset, Arena arena) throws Throwable {
            this.a = a;
            this.b = b;
            this.offset = offset;
            y = new double[ROWS + 8];
            heapY = MemorySegment.ofArray(y).asSlice(offset * 8L, ROWS * 8L);
            rootY = arena.allocate(ROWS * 8L + 4096, 4096);
            heapA = MemorySegment.ofArray(a).asSlice(offset * 8L, N * 8L);
            heapB = MemorySegment.ofArray(b).asSlice(offset * 8L, N * 8L);
            rootA = arena.allocate(N * 8L + 4096, 4096);
            rootB = arena.allocate(N * 8L + 4096, 4096);
            alignedA = arena.allocate(N * 8L, 64);
            alignedB = arena.allocate(N * 8L, 64);
            alignedA.copyFrom(heapA);
            alignedB.copyFrom(heapB);
            prepareNative();
        }

        void prepareNative() throws Throwable {
            long aOffset = pageOffset(heapA), bOffset = pageOffset(heapB);
            if (nativeA == null || (nativeA.address() & 4095) != aOffset) {
                nativeA = rootA.asSlice(aOffset, N * 8L);
                nativeA.copyFrom(heapA);
            }
            long yOffset = pageOffset(heapY);
            if (nativeY == null || (nativeY.address() & 4095) != yOffset) {
                nativeY = rootY.asSlice(yOffset, ROWS * 8L);
                nativeY.copyFrom(heapY);
            }
            if (nativeB == null || (nativeB.address() & 4095) != bOffset) {
                nativeB = rootB.asSlice(bOffset, N * 8L);
                nativeB.copyFrom(heapB);
            }
        }
    }

    static final class Kernel {
        final String name, operation;
        final MethodHandle handle;
        final boolean nativeCall;

        Kernel(String name, String operation, MethodHandle handle, boolean nativeCall) {
            this.name = name;
            this.operation = operation;
            this.handle = handle;
            this.nativeCall = nativeCall;
        }

        double call(Fixture f) throws Throwable {
            if (operation.equals("gemv")) {
                if (nativeCall) {
                    handle.invokeExact(102, 111, ROWS, ROWS, 1.0, f.nativeA, ROWS, f.nativeB, 1, 1.0, f.nativeY, 1);
                    return f.nativeY.get(JAVA_DOUBLE, 0L);
                }
                handle.invokeExact(1.0, f.a, f.offset, ROWS, f.b, f.offset, 1, ROWS, ROWS, f.y, f.offset, 1);
                return f.y[f.offset];
            }
            if (nativeCall) {
                MemorySegment nativeA = name.equals("onemkl-aligned") ? f.alignedA : f.nativeA;
                MemorySegment nativeB = name.equals("onemkl-aligned") ? f.alignedB : f.nativeB;
                return switch (operation) {
                    case "asum" -> (double) handle.invokeExact(N, nativeA, 1);
                    case "dot" -> (double) handle.invokeExact(N, nativeA, 1, nativeB, 1);
                    case "iamax" -> (long) handle.invokeExact(N, nativeA, 1);
                    default -> throw new AssertionError(operation);
                };
            }
            return switch (operation) {
                case "asum" -> (double) handle.invokeExact(f.a, f.offset, N);
                case "dot" -> (double) handle.invokeExact(f.a, f.offset, f.b, f.offset, N);
                case "iamax" -> (int) handle.invokeExact(f.a, f.offset, N);
                default -> throw new AssertionError(operation);
            };
        }
    }

    static double batch(Kernel kernel, Fixture fixture, int iterations) throws Throwable {
        double result = 0;
        for (int i = 0; i < iterations; i++) { double value = kernel.call(fixture); sink = value; result += value; }
        sink = result;
        return result;
    }

    static double reference(String operation, Fixture f) {
        if (operation.equals("gemv")) {
            double result = 0;
            for (int c = 0; c < ROWS; c++) result += f.a[f.offset + c * ROWS] * f.b[f.offset + c];
            return result;
        }
        double result = 0, best = 0;
        int winner = 0;
        for (int i = 0; i < N; i++) {
            double value = f.a[f.offset + i];
            if (operation.equals("dot")) result += value * f.b[f.offset + i];
            else if (operation.equals("asum")) result += Math.abs(value);
            else if (Math.abs(value) > best) {
                best = Math.abs(value);
                winner = i;
            }
        }
        return operation.equals("iamax") ? winner : result;
    }

    public static void main(String[] args) throws Throwable {
        Locale.setDefault(Locale.ROOT);
        Path directory = Path.of(args[0]).toAbsolutePath();
        var kernels = new ArrayList<Kernel>();
        var policies = new ArrayList<Policy>();
        String[] operations = {"asum", "iamax", "dot"};
        var dependencies = new ArrayList<URL>();
        for (String entry : Files.readString(Path.of(args[2])).trim().split(java.io.File.pathSeparator)) {
            Path path = Path.of(entry).toAbsolutePath();
            if (path.toString().endsWith(".jar") && !path.getFileName().toString().endsWith("koblas-jvm-SNAPSHOT.jar")) dependencies.add(path.toUri().toURL());
        }
        for (String pair : args[1].split(",")) {
            String[] parts = pair.split("=", 2);
            var urls = new ArrayList<URL>(dependencies);
            urls.add(Path.of(parts[1]).toAbsolutePath().toUri().toURL());
            var loader = new URLClassLoader(urls.toArray(URL[]::new), ClassLoader.getPlatformClassLoader());
            var type = Class.forName("com.eignex.koblas.internal.jvm25.SimdOps", true, loader);
            Object instance = type.getField("INSTANCE").get(null);
            System.err.println(parts[0]+" source="+type.getProtectionDomain().getCodeSource().getLocation());
            var hintType = Class.forName("com.eignex.koblas.internal.jvm25.HotSpotArrayAlignment", true, loader);
            Object hint = hintType.getField("INSTANCE").get(null);
            policies.add(new Policy(parts[0],
                MethodHandles.publicLookup().unreflect(hintType.getMethod("hasKnownAlignment", double[].class)).bindTo(hint),
                MethodHandles.publicLookup().unreflect(hintType.getMethod("prefix", double[].class, int.class, int.class)).bindTo(hint)));

            for (String operation : operations) {
                var method = operation.equals("dot") ? type.getMethod(operation, double[].class, int.class, double[].class, int.class, int.class)
                        : type.getMethod(operation, double[].class, int.class, int.class);
                kernels.add(new Kernel(parts[0], operation, MethodHandles.publicLookup().unreflect(method).bindTo(instance), false));
            }
        }
        var bean = (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory.getThreadMXBean();
        bean.setThreadAllocatedMemoryEnabled(true);
        long thread = Thread.currentThread().threadId();
        var collectors = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
        System.err.println("vm="+System.getProperty("java.vm.name")+" version="+Runtime.version()+" args="+java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments());
        System.err.println("collectors="+collectors.stream().map(c -> c.getName()).toList());
        System.err.println("preferred_lanes="+jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length());
        try (Arena arena = Arena.ofConfined()) {
            alignment = LINKER.downcallHandle(LINKER.defaultLookup().find("memchr").orElseThrow(),
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG), Linker.Option.critical(true));
            System.err.println("diagnostic_output=" + directory + " address_probe=libc/memchr");
            if (pageOffset(arena.allocate(8, 4096)) != 0) throw new AssertionError("alignment probe failed");
            var mkl = SymbolLookup.libraryLookup(Path.of(System.getenv("MKLROOT"), "lib/libmkl_rt.so.3"), arena);
            var version = arena.allocate(256);
            function(mkl, "MKL_Get_Version_String", FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT)).invokeExact(version, 256);
            System.err.println("oneMKL version="+version.getString(0));

            int interfaceResult = (int) function(mkl, "MKL_Set_Interface_Layer", FunctionDescriptor.of(JAVA_INT, JAVA_INT)).invokeExact(0);
            int threadingResult = (int) function(mkl, "MKL_Set_Threading_Layer", FunctionDescriptor.of(JAVA_INT, JAVA_INT)).invokeExact(1);
            if (interfaceResult < 0 || threadingResult < 0) throw new AssertionError("oneMKL configuration rejected");
            function(mkl, "MKL_Set_Num_Threads", FunctionDescriptor.ofVoid(JAVA_INT)).invokeExact(1);
            function(mkl, "MKL_Set_Dynamic", FunctionDescriptor.ofVoid(JAVA_INT)).invokeExact(0);
            kernels.add(new Kernel("onemkl-matched", "asum", function(mkl, "cblas_dasum", FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT)), true));
            kernels.add(new Kernel("onemkl-matched", "dot", function(mkl, "cblas_ddot", FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)), true));
            kernels.add(new Kernel("onemkl-matched", "iamax", function(mkl, "cblas_idamax", FunctionDescriptor.of(JAVA_LONG, JAVA_INT, ADDRESS, JAVA_INT)), true));
            for (var kernel : new ArrayList<Kernel>(kernels)) if (kernel.nativeCall) kernels.add(new Kernel("onemkl-aligned", kernel.operation, kernel.handle, true));
            int maxThreads = (int) function(mkl, "MKL_Get_Max_Threads", FunctionDescriptor.of(JAVA_INT)).invokeExact();
            if (maxThreads != 1) throw new AssertionError("oneMKL max threads=" + maxThreads);
            System.err.println("oneMKL configured max threads=" + maxThreads);
            double[] a = new double[N + 8], b = new double[N + 8];
            var random = new Random(934);
            for (int i = 0; i < a.length; i++) { a[i] = random.nextDouble() * 2 - 1; b[i] = random.nextDouble() * 2 - 1; }
            var fixtures = new ArrayList<Fixture>();
            for (int offset = 0; offset < 8; offset++) fixtures.add(new Fixture(a, b, offset, arena));
            int aHash = Arrays.hashCode(a), bHash = Arrays.hashCode(b);
            int width = jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorByteSize();
            for (var policy : policies) for (Fixture fixture : fixtures) {
                for (double[] values : new double[][] {a, b}) {
                    boolean known = (boolean) policy.known.invokeExact(values);
                    int actual = (int) policy.prefix.invokeExact(values, fixture.offset, N);
                    var segment = MemorySegment.ofArray(values).asSlice(fixture.offset * 8L, N * 8L);
                    int expected = (int) (-pageOffset(segment) & (width - 1)) / 8;
                    if ((known && actual != expected) || (!known && actual != 0)) {
                        throw new AssertionError("Production prefix disagrees with physical layout");
                    }
                    System.err.println("alignment_policy arm=" + policy.name + " offset=" + fixture.offset
                        + " known=" + known + " prefix=" + actual + " physical_prefix=" + expected);
                }
            }

            for (Kernel kernel : kernels) {
                for (Fixture f : fixtures) {
                    Arrays.fill(f.y, 0.0);
                    f.nativeY.fill((byte) 0);
                    double expected = reference(kernel.operation, f), actual = kernel.call(f);
                    if (Math.abs(expected - actual) > 1e-10 * Math.max(1, Math.abs(expected))) {
                        throw new AssertionError(kernel.name + " " + kernel.operation + " reference=" + expected + " actual=" + actual);
                    }
                }
                long deadline = System.nanoTime() + Long.getLong("trial.warmupNanos", 600_000_000L);
                while (System.nanoTime() < deadline) batch(kernel, fixtures.getFirst(), 1000);
            }
            System.out.println("round,sample,operation,arm,offset,iterations,elapsed_ns,ns_per_call,heap_a_before,heap_a_after,heap_b_before,heap_b_after,native_a,native_b,heap_y_before,heap_y_after,native_y,checksum,bytes_per_call,gc_events,prefix_a,prefix_b");
            for (int round = 0; round < ROUNDS; round++) {
                Collections.rotate(fixtures, 3);
                for (Fixture f : fixtures) {
                    for (String operation : operations) {
                        var selected = new ArrayList<Kernel>(kernels.stream().filter(k -> k.operation.equals(operation)).toList());
                        for (int sample = 0; sample < SAMPLES; sample++) {
                            Collections.rotate(selected, 1);
                            for (Kernel kernel : selected) {
                                if (kernel.nativeCall) f.prepareNative();
                                long aBefore = pageOffset(f.heapA), bBefore = pageOffset(f.heapB), yBefore = pageOffset(f.heapY);
                                int vectorBytes = jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorByteSize();
                                int prefixA = (int)(-aBefore & (vectorBytes - 1)) / 8, prefixB = (int)(-bBefore & (vectorBytes - 1)) / 8;
                                long gcBefore = 0;
                                for (var collector : collectors) gcBefore += Math.max(0, collector.getCollectionCount());
                                long allocatedBefore = bean.getThreadAllocatedBytes(thread);
                                long start = System.nanoTime();
                                double checksum = batch(kernel, f, ITERATIONS);
                                long elapsed = System.nanoTime() - start;
                                long bytes = bean.getThreadAllocatedBytes(thread) - allocatedBefore;
                                long gcAfter = 0;
                                for (var collector : collectors) gcAfter += Math.max(0, collector.getCollectionCount());
                                long aAfter = pageOffset(f.heapA), bAfter = pageOffset(f.heapB), yAfter = pageOffset(f.heapY);
                                System.out.printf("%d,%d,%s,%s,%d,%d,%d,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.12f,%.6f,%d,%d,%d%n",
                                        round + 1, sample + 1, operation, kernel.name, f.offset, ITERATIONS, elapsed,
                                        elapsed / (double) ITERATIONS, aBefore, aAfter, bBefore, bAfter,
                                        (kernel.name.equals("onemkl-aligned") ? f.alignedA : f.nativeA).address() & 4095, (kernel.name.equals("onemkl-aligned") ? f.alignedB : f.nativeB).address() & 4095, yBefore, yAfter,
                                        f.nativeY.address() & 4095, checksum, bytes / (double) ITERATIONS, gcAfter - gcBefore, prefixA, prefixB);
                            }
                        }
                    }
                }
                if (Arrays.hashCode(a) != aHash || Arrays.hashCode(b) != bHash) throw new AssertionError("input changed");
            }
        }
    }
}
