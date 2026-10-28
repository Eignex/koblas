"""Verify the JVM publication's bytecode boundary and consumer compatibility metadata."""

import json
import struct
import sys
import zipfile
from pathlib import Path

jar = Path(sys.argv[1])
metadata = Path(sys.argv[2])
with zipfile.ZipFile(jar) as archive:
    assert b"Multi-Release: true" in archive.read("META-INF/MANIFEST.MF")
    base_count = versioned_count = 0
    for name in archive.namelist():
        assert not name.endswith((".so", ".a", ".dll", ".dylib")), name
        if not name.endswith(".class"):
            continue
        data = archive.read(name)
        major = struct.unpack_from(">H", data, 6)[0]
        if name.startswith("META-INF/versions/25/"):
            assert major == 69, (name, major)
            versioned_count += 1
        else:
            assert not name.startswith("META-INF/versions/") or name == "META-INF/versions/17/module-info.class", name
            assert major == 61, (name, major)
            for forbidden in (b"java/lang/foreign/", b"jdk/incubator/vector/", b"java/lang/management/", b"com/sun/management/"):
                assert forbidden not in data, (name, forbidden)
            base_count += 1
    assert "com/eignex/koblas/JvmBackend.class" in archive.namelist()
    assert "META-INF/versions/25/com/eignex/koblas/JvmBackend.class" in archive.namelist()
    assert base_count and versioned_count

variants = json.loads(metadata.read_text())["variants"]
for variant in variants:
    if variant["attributes"].get("org.gradle.category") == "library":
        assert variant["attributes"]["org.gradle.jvm.version"] == 17, variant
print(f"Verified {base_count} base and {versioned_count} desktop classes; published JVM baseline is 17")

for apk_path in sys.argv[3:]:
    with zipfile.ZipFile(apk_path) as archive:
        for name in archive.namelist():
            assert not name.startswith("lib/"), (apk_path, name)
            if name.endswith(".dex"):
                data = archive.read(name)
                for forbidden in (b"java/lang/foreign/", b"jdk/incubator/vector/"):
                    assert forbidden not in data, (apk_path, name, forbidden)
    print(f"Verified portable Android APK without native libraries: {apk_path}")
