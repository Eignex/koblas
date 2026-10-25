# Instrumentation reaches the consumer entry point from the separate test APK.
-keep class com.eignex.koblas.consumer.ConsumerCheck { public *; }
# The separate test APK shares the app's stdlib; its runner calls are invisible to the app shrinker.
-keep class kotlin.** { *; }
