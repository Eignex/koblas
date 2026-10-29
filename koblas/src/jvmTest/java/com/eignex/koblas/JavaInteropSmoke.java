package com.eignex.koblas;

final class JavaInteropSmoke {
    private JavaInteropSmoke() {}

    static void run() {
        DenseMatrix matrix = DenseMatrix.ofRows(new double[][] {
            {2.0, 1.0},
            {1.0, 3.0},
        });
        DenseMatrix identity = DenseMatrix.diagonal(2);
        DenseVector vector = DenseVector.of(new double[] {3.0, 5.0});

        DenseMatrix product = Koblas.multiply(matrix, identity);
        DenseVector result = Koblas.multiply(product, vector);
        assertArrayEquals(new double[] {11.0, 18.0}, result.toDoubleArray());
        assertEquals(34.0, Koblas.dot(vector, vector));

        double[] destination = new double[2];
        Koblas.gemvInto(matrix, vector, destination);
        assertArrayEquals(result.toDoubleArray(), destination);
        assertEquals(8.0, Koblas.asum(vector));

        StridedVector view = Koblas.asView(vector);
        Koblas.scale(view, 2.0);
        assertArrayEquals(new double[] {6.0, 10.0}, vector.toDoubleArray());

        SparseMatrix sparse = SparseMatrix.ofTriplets(
            2,
            2,
            new int[] {0, 1},
            new int[] {0, 1},
            new double[] {2.0, 3.0}
        );
        double[] scattered = new double[2];
        Koblas.getDefault().getSparseKernels().scatter(Koblas.column(sparse, 0), scattered);
        assertArrayEquals(new double[] {2.0, 0.0}, scattered);

        PreparedSparseMatrix prepared = Koblas.prepare(sparse);
        double[] preparedResult = new double[2];
        prepared.gemvInto(1.0, new double[] {4.0, 5.0}, 0.0, preparedResult);
        assertArrayEquals(new double[] {8.0, 15.0}, preparedResult);
        assertArrayEquals(new double[] {8.0, 15.0}, Koblas.gemv(sparse, new double[] {4.0, 5.0}));

        KoblasEngine engine = BuiltinEngines.getScalar();
        DenseMatrixRegion outputRegion = DenseMatrix.zero(5, 5).view(1, 1, 2, 2);
        engine.gemm(1.0, (Matrix) prepared, false, (Matrix) identity, false, 0.0, outputRegion, new Workspace());
        assertEquals(2.0, outputRegion.get(0, 0));
        assertEquals(3.0, outputRegion.get(1, 1));
        Koblas.trmm(prepared, outputRegion, true, false, true, false, 1.0, new Workspace());
        assertEquals(2.0, outputRegion.get(0, 0));

        DenseVector reverseInput = new StridedVector(new double[] {5.0, 0.0, 3.0}, 2, 2, -2);
        double[] outputBuffer = new double[] {-7.0, -7.0, -7.0, -7.0, -7.0};
        DenseVector reverseOutput = new StridedVector(outputBuffer, 3, 2, -2);
        engine.gemv(1.0, matrix, reverseInput, 0.0, reverseOutput, false, null);
        assertArrayEquals(new double[] {-7.0, 18.0, -7.0, 11.0, -7.0}, outputBuffer);

        DenseVector rotatedX = DenseVector.of(new double[] {3.0});
        DenseVector rotatedY = DenseVector.of(new double[] {4.0});
        // The rotation that takes (3, 4) to (5, 0); only the first is exact in binary, and this asserts bits.
        Koblas.rot(rotatedX, rotatedY, 0.6, 0.8);
        assertEquals(5.0, rotatedX.get(0));

        if (Koblas.getDefault() == null || BuiltinEngines.getScalar() == null) {
            throw new AssertionError("Java engine accessors must return engines");
        }
    }

    private static void assertArrayEquals(double[] expected, double[] actual) {
        if (expected.length != actual.length) {
            throw new AssertionError("array lengths differ");
        }
        for (int index = 0; index < expected.length; index++) {
            assertEquals(expected[index], actual[index]);
        }
    }

    private static void assertEquals(double expected, double actual) {
        if (Double.doubleToLongBits(expected) != Double.doubleToLongBits(actual)) {
            throw new AssertionError("expected " + expected + ", got " + actual);
        }
    }
}
