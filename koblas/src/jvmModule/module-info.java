/** Desktop implementation module; public API packages match the portable JVM artifact. */
module com.eignex.koblas {
    requires transitive kotlin.stdlib;
    requires transitive kotlinx.serialization.core;
    requires java.management;
    requires jdk.management;
    requires static jdk.incubator.vector;

    exports com.eignex.koblas;
    exports com.eignex.koblas.dense;
    exports com.eignex.koblas.sparse;
    exports com.eignex.koblas.vendor;
}
