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
        assertEquals(4.0, Koblas.normInf(matrix));

        StridedVectorView view = Koblas.asView(vector);
        Koblas.scale(view, 2.0);
        assertArrayEquals(new double[] {6.0, 10.0}, vector.toDoubleArray());

        SparseMatrix sparse = SparseMatrix.ofTriplets(
            2,
            2,
            new int[] {0, 1},
            new int[] {0, 1},
            new double[] {2.0, 3.0}
        );
        PreparedSparseMatrix prepared = Koblas.prepare(sparse);
        double[] preparedResult = new double[2];
        prepared.gemv(1.0, new double[] {4.0, 5.0}, 0.0, preparedResult);
        assertArrayEquals(new double[] {8.0, 15.0}, preparedResult);

        Givens rotation = Koblas.rotg(3.0, 4.0);
        assertEquals(5.0, rotation.getR());

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
