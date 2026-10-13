#!/usr/bin/env python3
"""Regression checks for release version selection and the bundle accepted by the upload step."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("release_bundle", SCRIPTS / "check-release-bundle.py")
bundle_checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle_checker)


class ResolveReleaseVersionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "-q")
        self.git("-c", "user.name=CI", "-c", "user.email=ci@example.invalid",
                 "commit", "--allow-empty", "-qm", "test: create version fixture")
        self.git("tag", "v0.1.0")
        self.git("tag", "v9.0.0-rc.1")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)

    def resolve(self, ref_type="branch", ref_name="main", dry_run="false"):
        output = self.root / "outputs"
        output.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", str(SCRIPTS / "resolve-release-version.sh")], cwd=self.root, capture_output=True,
            env={**os.environ, "GITHUB_OUTPUT": str(output), "GITHUB_REF_TYPE": ref_type,
                 "GITHUB_REF_NAME": ref_name, "DRY_RUN": dry_run},
        )
        if result.returncode:
            return result, {}
        return result, dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_manual_snapshot_uses_next_stable_patch(self):
        result, outputs = self.resolve()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(outputs["version"], "0.1.1-SNAPSHOT")
        self.assertEqual(outputs["is_snapshot"], "true")
        self.assertEqual(outputs["dry_run"], "false")
        self.assertEqual(outputs["revision"], self.git("rev-parse", "HEAD").stdout.decode().strip())

    def test_tag_uses_exact_prerelease_version(self):
        _, outputs = self.resolve("tag", "v0.2.0-rc.1")
        self.assertEqual(outputs["version"], "0.2.0-rc.1")
        self.assertEqual(outputs["is_snapshot"], "false")

    def test_dry_run_on_tag_remains_a_dry_run(self):
        _, outputs = self.resolve("tag", "v0.2.0", "true")
        self.assertEqual(outputs["version"], "0.1.1")
        self.assertEqual(outputs["is_snapshot"], "false")
        self.assertEqual(outputs["dry_run"], "true")

    def test_repository_without_stable_tags_starts_at_first_patch(self):
        self.git("tag", "-d", "v0.1.0")
        _, outputs = self.resolve()
        self.assertEqual(outputs["version"], "0.0.1-SNAPSHOT")

    def test_malformed_release_tag_is_rejected(self):
        result, _ = self.resolve("tag", "vnext")
        self.assertNotEqual(result.returncode, 0)


class CheckReleaseBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.gpg = ["gpg", "--homedir", str(cls.root), "--batch", "--pinentry-mode", "loopback", "--passphrase", ""]
        subprocess.run(cls.gpg + ["--quick-generate-key", "CI <ci@example.invalid>", "rsa2048", "sign", "0"],
                       check=True, capture_output=True)
        cls.key = subprocess.run(cls.gpg + ["--armor", "--export-secret-keys"], check=True,
                                 capture_output=True).stdout.decode()
        cls.entries = {}
        for artifact, extension in bundle_checker.ARTIFACTS.items():
            base = f"com/eignex/{artifact}/0.1.1/{artifact}-0.1.1"
            pom = f'''<project xmlns="http://maven.apache.org/POM/4.0.0">
              <groupId>com.eignex</groupId><artifactId>{artifact}</artifactId><version>0.1.1</version>
              <name>Koblas</name><description>BLAS</description><url>https://example.invalid</url>
              <licenses><license><name>Apache-2.0</name><url>https://example.invalid/license</url></license></licenses>
              <developers><developer><id>ci</id></developer></developers>
              <scm><url>https://example.invalid</url><connection>scm:git:https://example.invalid</connection></scm>
            </project>'''.encode()
            component = {"group": "com.eignex", "module": "koblas", "version": "0.1.1"}
            if artifact != "koblas":
                component["url"] = "../../koblas/0.1.1/koblas-0.1.1.module"
            module = json.dumps({"component": component}).encode()
            payloads = {".pom": pom, ".module": module, f".{extension}": b"binary",
                        "-sources.jar": b"sources", "-javadoc.jar": b"docs"}
            if artifact == "koblas":
                payloads["-kotlin-tooling-metadata.json"] = b"{}"
            if artifact == "koblas-macosarm64":
                payloads["-metadata.jar"] = b"metadata"
            for suffix, data in payloads.items():
                cls.entries[base + suffix] = data
                cls.entries[base + suffix + ".asc"] = subprocess.run(
                    cls.gpg + ["--armor", "--detach-sign"], input=data, check=True, capture_output=True,
                ).stdout

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["gpgconf", "--homedir", str(cls.root), "--kill", "all"], capture_output=True)
        cls.directory.cleanup()

    def verify(self, entries, version="0.1.1"):
        path = self.root / "bundle.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for name, data in entries.items():
                bundle.writestr(name, data)
        with contextlib.redirect_stdout(io.StringIO()):
            bundle_checker.check_bundle(path, version, self.key)

    def test_complete_signed_bundle_is_accepted(self):
        self.verify(self.entries)

    def test_benchmark_publication_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected publication"):
            self.verify({**self.entries, "com/eignex/koblas-bench/0.1.1/koblas-bench-0.1.1.pom": b"benchmark"})

    def test_missing_host_built_macos_binary_is_rejected(self):
        entries = {name: data for name, data in self.entries.items() if not name.endswith(".klib")}
        with self.assertRaisesRegex(ValueError, "missing artifact"):
            self.verify(entries)

    def test_wrong_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wrong publication version"):
            self.verify(self.entries, "0.1.2")

    def test_incomplete_publication_metadata_is_rejected(self):
        entries = dict(self.entries)
        name = "com/eignex/koblas-jvm/0.1.1/koblas-jvm-0.1.1.pom"
        entries[name] = entries[name].replace(b"<description>BLAS</description>", b"")
        with self.assertRaisesRegex(ValueError, "missing POM description"):
            self.verify(entries)

    def test_unsigned_payload_is_rejected(self):
        entries = dict(self.entries)
        del entries["com/eignex/koblas-jvm/0.1.1/koblas-jvm-0.1.1.jar.asc"]
        with self.assertRaisesRegex(ValueError, "missing signature"):
            self.verify(entries)

    def test_corrupted_payload_is_rejected(self):
        entries = dict(self.entries)
        entries["com/eignex/koblas-jvm/0.1.1/koblas-jvm-0.1.1.jar"] = b"corrupted"
        with self.assertRaisesRegex(ValueError, "invalid signature"):
            self.verify(entries)

    def test_invalid_checksum_is_rejected(self):
        entries = {**self.entries, "com/eignex/koblas-jvm/0.1.1/koblas-jvm-0.1.1.jar.sha256": b"wrong"}
        with self.assertRaisesRegex(ValueError, "invalid checksum"):
            self.verify(entries)


if __name__ == "__main__":
    unittest.main()
