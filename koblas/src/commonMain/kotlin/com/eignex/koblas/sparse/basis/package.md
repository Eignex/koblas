# Package com.eignex.koblas.sparse.basis

This package contains the indexed-vector and basis-solver contracts required by the HFactor host backend.
There is no portable basis implementation. The package is isolated from sparse BLAS so removing HFactor
later does not disturb sparse matrix arithmetic.
