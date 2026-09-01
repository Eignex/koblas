import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "check-benchmark-coverage.py"
SPEC = importlib.util.spec_from_file_location("benchmark_coverage", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckBenchmarkCoverageTest(unittest.TestCase):
    def write(self, root, relative, text):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def manifest(self, root, rows):
        return self.write(
            root,
            "benchmark-coverage.tsv",
            "api_signature\tbenchmark_id\tstatus\tnotes\n" + "\n".join(rows) + "\n",
        )

    def inventory(self, root, rows):
        return self.write(
            root,
            "public-numerical-api.tsv",
            "api_surface\tapi_signature\n" + "\n".join(rows) + "\n",
        )

    def test_inventory_covers_each_public_operation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = self.write(
                root,
                "dense/F64Blas.kt",
                "public interface F64Blas { public fun gemv() }\n",
            )
            manifest = self.manifest(root, ["F64Blas.gemv\t\texcluded\tSmall fixture has no benchmark source."])
            inventory = self.inventory(root, ["dense/F64Blas.kt:F64Blas.gemv()\tF64Blas.gemv"])
            original = CHECKER.PUBLIC_NUMERICAL_SOURCES
            self.addCleanup(setattr, CHECKER, "PUBLIC_NUMERICAL_SOURCES", original)
            CHECKER.PUBLIC_NUMERICAL_SOURCES = ("dense/F64Blas.kt",)

            signatures, _, _ = CHECKER.manifest(manifest)
            CHECKER.api_inventory(inventory, signatures, CHECKER.public_numerical_operations(root))
            self.assertEqual(source.name, "F64Blas.kt")

    def test_inventory_rejects_a_new_unreviewed_public_operation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write(
                root,
                "dense/F64Blas.kt",
                "public interface F64Blas { public fun gemv(); public fun geam() }\n",
            )
            manifest = self.manifest(root, ["F64Blas.gemv\t\texcluded\tSmall fixture has no benchmark source."])
            inventory = self.inventory(root, ["dense/F64Blas.kt:F64Blas.gemv()\tF64Blas.gemv"])
            original = CHECKER.PUBLIC_NUMERICAL_SOURCES
            self.addCleanup(setattr, CHECKER, "PUBLIC_NUMERICAL_SOURCES", original)
            CHECKER.PUBLIC_NUMERICAL_SOURCES = ("dense/F64Blas.kt",)

            signatures, _, _ = CHECKER.manifest(manifest)
            with self.assertRaisesRegex(SystemExit, "absent from inventory.*geam"):
                CHECKER.api_inventory(inventory, signatures, CHECKER.public_numerical_operations(root))

    def test_inventory_rejects_an_unreviewed_overload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write(
                root,
                "dense/F64Blas.kt",
                "public interface F64Blas {\n"
                "    public fun gemv(a: F64DenseMatrix)\n"
                "    public fun gemv(a: F64SparseMatrix)\n"
                "}\n",
            )
            manifest = self.manifest(root, ["F64Blas.gemv\t\texcluded\tSmall fixture has no benchmark source."])
            inventory = self.inventory(root, ["dense/F64Blas.kt:F64Blas.gemv(F64DenseMatrix)\tF64Blas.gemv"])
            original = CHECKER.PUBLIC_NUMERICAL_SOURCES
            self.addCleanup(setattr, CHECKER, "PUBLIC_NUMERICAL_SOURCES", original)
            CHECKER.PUBLIC_NUMERICAL_SOURCES = ("dense/F64Blas.kt",)

            signatures, _, _ = CHECKER.manifest(manifest)
            with self.assertRaisesRegex(SystemExit, r"absent from inventory.*gemv\(F64SparseMatrix\)"):
                CHECKER.api_inventory(inventory, signatures, CHECKER.public_numerical_operations(root))

    def test_a_comment_is_not_a_public_declaration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write(
                root,
                "dense/F64Blas.kt",
                "/** Prefer [gemv] over `public fun geam()` here. */\n"
                "public interface F64Blas { public fun gemv() }\n",
            )
            original = CHECKER.PUBLIC_NUMERICAL_SOURCES
            self.addCleanup(setattr, CHECKER, "PUBLIC_NUMERICAL_SOURCES", original)
            CHECKER.PUBLIC_NUMERICAL_SOURCES = ("dense/F64Blas.kt",)

            self.assertEqual(
                CHECKER.public_numerical_operations(root),
                {"dense/F64Blas.kt:F64Blas.gemv()"},
            )

    def test_exclusion_requires_a_reason_and_no_benchmark_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.manifest(pathlib.Path(temporary), ["F64Blas.gemv\tmethod\texcluded\t"])
            with self.assertRaisesRegex(SystemExit, "needs notes and no benchmark method"):
                CHECKER.manifest(manifest)


if __name__ == "__main__":
    unittest.main()
