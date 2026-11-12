# Internal JVM assertion policy

Build the experimental packaged JVM artifact with:

```bash
./gradlew check lintDocs -Pkoblas.stripInternalAssertions=true
```

The default build keeps assertions. This opt-in policy runs on `jvmJar`, so publication,
packaged tests and benchmark preparation consume the same transformed artifact. It uses
JDK 25's Class-File API at build time; the baseline classes still target Java 17 and the
tool adds no runtime dependency. The policy and tool classpath participate in the JAR's
Gradle cache inputs.

Inside `com.eignex.koblas.internal`, it removes generated parameter and expression null
assertion calls, trusting Kotlin's declared non-null contracts. It also removes a cast
null assertion immediately after a Vector API `reinterpretShape` result, whose contract
guarantees a non-null vector. Both the stack and local-variable compiler forms are
recognized, with no alternate control-flow entry into the sequence. Unknown producers
retain their assertions. Argument evaluation remains intact.

Public API classes remain byte-for-byte intact. Kotlin metadata, explicit validation,
other `checkNotNull` calls, JVM casts, array bounds checks and numerical instructions
are retained. Kotlin `internal` declarations are public in JVM bytecode; Java callers
of implementation packages must obey the implementation's non-null contracts.

Every output class is verified before the JAR is replaced. The fixture suite checks
public assertions, argument side effects, explicit null failures, vector producer
provenance, alternate entries, wrong-type casts and packaged metadata. Repository tests
exercise the packaged artifact on Java 17 and Java 25. Allocation checks also need the
three SIMD runtime profiles: normal, FMA disabled, and narrow vectors with FMA disabled.

This is a performance experiment. Compare identical numerical sources with the property
disabled and enabled, using the benchmark suite's paired protocol and unchanged Native
arms as controls. A smaller assertion count alone is not performance evidence.
