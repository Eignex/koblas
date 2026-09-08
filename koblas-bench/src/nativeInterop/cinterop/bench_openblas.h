#pragma once

double cblas_ddot(int, const double *, int, const double *, int);
void cblas_daxpy(int, double, const double *, int, double *, int);
void cblas_dscal(int, double, double *, int);
double cblas_dnrm2(int, const double *, int);
double cblas_dasum(int, const double *, int);
void cblas_dswap(int, double *, int, double *, int);
void cblas_drotm(int, double *, int, double *, int, const double *);
void cblas_drot(int, double *, int, double *, int, double, double);
void cblas_dgemv(int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
void cblas_dsymv(int, int, int, double, const double *, int, const double *, int, double, double *, int);
void cblas_dger(int, int, int, double, const double *, int, const double *, int, double *, int);
void cblas_dsyr(int, int, int, double, const double *, int, double *, int);
void cblas_dsyr2(int, int, int, double, const double *, int, const double *, int, double *, int);
void cblas_dtrsv(int, int, int, int, int, const double *, int, double *, int);
void cblas_dtrmv(int, int, int, int, int, const double *, int, double *, int);
void cblas_dgemm(int, int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
void cblas_dsyrk(int, int, int, int, int, double, const double *, int, double, double *, int);
void cblas_dsyr2k(int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
void cblas_dsymm(int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
void cblas_dtrsm(int, int, int, int, int, int, int, double, const double *, int, double *, int);
void cblas_dtrmm(int, int, int, int, int, int, int, double, const double *, int, double *, int);
void openblas_set_num_threads(int);
