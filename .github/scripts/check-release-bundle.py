#!/usr/bin/env python3
"""Check publication coordinates, required artifacts, checksums and detached signatures before upload."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile


ARTIFACTS = {
    "koblas": "jar",
    "koblas-jvm": "jar",
    "koblas-linuxx64": "klib",
    "koblas-linuxarm64": "klib",
    "koblas-macosarm64": "klib",
}
CHECKSUMS = ("md5", "sha1", "sha256", "sha512")


def check_bundle(path, version, signing_key):
    with zipfile.ZipFile(path) as bundle:
        entries = [entry.filename for entry in bundle.infolist() if not entry.is_dir()]
        if len(entries) != len(set(entries)):
            raise ValueError("duplicate bundle entries")
        payloads = {}
        for name in entries:
            parts = name.split("/")
            if len(parts) != 5 or parts[:2] != ["com", "eignex"] or parts[2] not in ARTIFACTS:
                raise ValueError(f"unexpected publication: {name}")
            if parts[3] != version:
                raise ValueError(f"wrong publication version: {name}")
            artifact = parts[2]
            base = f"{artifact}-{version}"
            filenames = {f"{base}.{ext}" for ext in ("pom", "module", ARTIFACTS[artifact])}
            filenames.update((f"{base}-sources.jar", f"{base}-javadoc.jar"))
            if artifact == "koblas":
                filenames.add(f"{base}-kotlin-tooling-metadata.json")
            if ARTIFACTS[artifact] == "klib":
                filenames.add(f"{base}-metadata.jar")
            filename = parts[4]
            for suffix in CHECKSUMS:
                if filename.endswith(f".{suffix}"):
                    filename = filename[: -(len(suffix) + 1)]
                    break
            if filename.endswith(".asc"):
                filename = filename[:-4]
            if filename not in filenames:
                raise ValueError(f"unexpected artifact: {name}")
            if parts[4] in filenames:
                payloads[name] = bundle.read(name)

        for artifact, extension in ARTIFACTS.items():
            base = f"com/eignex/{artifact}/{version}/{artifact}-{version}"
            for suffix in (f".{extension}", ".pom", ".module", "-sources.jar", "-javadoc.jar"):
                name = base + suffix
                if name not in payloads:
                    raise ValueError(f"missing artifact: {name}")
            pom = ET.fromstring(payloads[base + ".pom"])
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}
            for field, expected in (("groupId", "com.eignex"), ("artifactId", artifact), ("version", version)):
                if pom.findtext(f"m:{field}", namespaces=ns) != expected:
                    raise ValueError(f"wrong POM {field}: {artifact}")
            for field in ("name", "description", "url", "licenses/license/name", "licenses/license/url",
                          "developers/developer/id", "scm/url", "scm/connection"):
                if not pom.findtext("/".join(f"m:{part}" for part in field.split("/")), namespaces=ns):
                    raise ValueError(f"missing POM {field}: {artifact}")
            component = json.loads(payloads[base + ".module"])["component"]
            if (component["group"], component["module"], component["version"]) != ("com.eignex", artifact, version):
                raise ValueError(f"wrong module coordinates: {artifact}")

        for name in payloads:
            if name + ".asc" not in entries:
                raise ValueError(f"missing signature: {name}")
        for name in entries:
            if name.endswith(".asc") and name[:-4] not in payloads:
                raise ValueError(f"orphan signature: {name}")
            for suffix in CHECKSUMS:
                if name.endswith(f".{suffix}"):
                    target = name[: -(len(suffix) + 1)]
                    if target not in entries:
                        raise ValueError(f"orphan checksum: {name}")
                    if hashlib.new(suffix, bundle.read(target)).hexdigest() != bundle.read(name).decode().strip():
                        raise ValueError(f"invalid checksum: {name}")
                    break

        if not signing_key:
            raise ValueError("SIGNING_KEY is required to verify the bundle")
        with tempfile.TemporaryDirectory() as directory:
            command = ["gpg", "--homedir", directory, "--batch"]
            imported = subprocess.run(command + ["--import"], input=signing_key.encode(), capture_output=True)
            if imported.returncode:
                raise ValueError("could not import the release signing key")
            for index, (name, data) in enumerate(payloads.items()):
                payload = Path(directory) / f"payload-{index}"
                signature = payload.with_suffix(".asc")
                payload.write_bytes(data)
                signature.write_bytes(bundle.read(name + ".asc"))
                verified = subprocess.run(command + ["--verify", str(signature), str(payload)], capture_output=True)
                if verified.returncode:
                    raise ValueError(f"invalid signature: {name}")
    print(f"Verified {len(payloads)} signed artifacts across {len(ARTIFACTS)} publications at {version}")


if __name__ == "__main__":
    try:
        check_bundle(sys.argv[1], sys.argv[2], os.environ.get("SIGNING_KEY"))
    except (ValueError, KeyError, ET.ParseError, zipfile.BadZipFile) as error:
        sys.exit(f"::error::{error}")
