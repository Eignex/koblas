#define _POSIX_C_SOURCE 200809L
#define WORKLOAD_VERSION "3"
#define FIXTURE_VERSION "1"
#include <cblas.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_MKL
typedef void *sparse_matrix_t;
typedef struct { int type; int mode; int diag; } matrix_descr;
extern void MKL_Set_Num_Threads(int);
extern void MKL_Set_Dynamic(int);
extern void MKL_Get_Version_String(char *, int);
extern int mkl_sparse_d_create_csr(sparse_matrix_t *, int, int, int, int *, int *, int *, double *);
extern int mkl_sparse_destroy(sparse_matrix_t);
extern int mkl_sparse_optimize(sparse_matrix_t);
extern int mkl_sparse_d_mv(int, double, sparse_matrix_t, matrix_descr, const double *, double, double *);
extern int mkl_sparse_d_mm(int, double, sparse_matrix_t, matrix_descr, int, const double *, int, int, double, double *, int);
extern int mkl_sparse_d_trsv(int, double, sparse_matrix_t, matrix_descr, const double *, double *);
extern int mkl_sparse_d_trsm(int, double, sparse_matrix_t, matrix_descr, int, const double *, int, int, double *, int);
extern int mkl_sparse_spmm(int, sparse_matrix_t, sparse_matrix_t, sparse_matrix_t *);
extern int mkl_sparse_d_syrkd(int, sparse_matrix_t, double, double, double *, int, int);
extern int mkl_sparse_syrk(int, sparse_matrix_t, sparse_matrix_t *);
extern int mkl_sparse_d_add(int, sparse_matrix_t, double, sparse_matrix_t, sparse_matrix_t *);
extern double cblas_ddoti(int, const double *, const int *, const double *);
extern void cblas_daxpyi(int, double, const double *, const int *, double *);
extern void cblas_dsctr(int, const double *, const int *, double *);
extern void cblas_dgthr(int, const double *, double *, const int *);
extern void cblas_dgthrz(int, double *, double *, const int *);
#else
extern char *openblas_get_config(void);
extern void openblas_set_num_threads(int);
extern double cblas_dsum(const int, const double *, const int);
#endif
extern void cblas_dgemmt(const CBLAS_LAYOUT, const CBLAS_UPLO, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE, const int, const int, const double, const double *, const int, const double *, const int, const double, double *, const int);

enum { MAX_DIMS = 3, MAX_OPTIONS = 8 };
static const uint64_t SEED = UINT64_C(0x243f6a8885a308d3);
static const uint64_t GOLDEN = UINT64_C(0x9e3779b97f4a7c15);

typedef struct {
    char id[256], operation[48], fixture[32];
    int dims[MAX_DIMS], dim_count;
    char option_names[MAX_OPTIONS][16], option_values[MAX_OPTIONS][32];
    int option_count;
} bench_case;

typedef struct {
    uint64_t state;
} portable_random;

typedef struct {
    int rows, cols, nnz;
    int *col_ptr, *row_idx;
    double *values;
} sparse_fixture;

typedef struct work work;
struct work {
    bench_case *spec;
    double *a, *b, *c, *initial, *x, *y;
    int *indices;
    sparse_fixture sa, sb;
#ifdef USE_MKL
    sparse_fixture csr_a, csr_b;
    sparse_matrix_t handle_a, handle_b;
#endif
    int supported;
    const char *comparison, *timing;
    double (*invoke)(work *);
};

static void fail(const char *message) { fprintf(stderr, "error: %s\n", message); exit(2); }
static void *allocate(size_t count, size_t size) { void *p = calloc(count ? count : 1, size); if (!p) fail("out of memory"); return p; }
static uint64_t mix64(uint64_t z) { z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9); z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb); return z ^ (z >> 31); }
static uint64_t random_long(portable_random *r) { r->state += GOLDEN; return mix64(r->state); }
static double random_double(portable_random *r) { return (double)(random_long(r) >> 11) * 2.220446049250313e-16 - 1.0; }
static portable_random stream(int operand) { portable_random r = { SEED ^ ((uint64_t)operand * GOLDEN) }; return r; }
static void fill_vector(double *out, int count, int operand) { portable_random r = stream(operand); for (int i = 0; i < count; ++i) out[i] = random_double(&r); }
static uint64_t digest(const double *values, int count) { uint64_t h = UINT64_C(14695981039346656037); for (int i = 0; i < count; ++i) { uint64_t bits; memcpy(&bits, values + i, 8); for (int b = 0; b < 8; ++b) { h = (h ^ (bits & 255)) * UINT64_C(1099511628211); bits >>= 8; } } return h; }

typedef struct { int64_t key; int row; } candidate;
static int compare_candidate(const void *left, const void *right) { const candidate *a = left, *b = right; if (a->key < b->key) return -1; if (a->key > b->key) return 1; return a->row - b->row; }
static int compare_int(const void *left, const void *right) { int a = *(const int *)left, b = *(const int *)right; return (a > b) - (a < b); }

static sparse_fixture make_sparse(int rows, int cols, double density, int operand, int triangular, int lower) {
    sparse_fixture out = { rows, cols, 0, allocate((size_t)cols + 1, sizeof(int)), NULL, NULL };
    int count = (int)(rows * density + 0.5); if (count < 1) count = 1; if (count > rows) count = rows;
    int capacity = cols * (count + (triangular ? 1 : 0));
    out.row_idx = allocate(capacity, sizeof(int)); out.values = allocate(capacity, sizeof(double));
    candidate *candidates = allocate(rows, sizeof(candidate)); int *selected = allocate(count, sizeof(int));
    for (int j = 0; j < cols; ++j) {
        uint64_t column_seed = SEED ^ ((uint64_t)((int64_t)operand * 65537 + j) * GOLDEN);
        for (int row = 0; row < rows; ++row) { candidates[row].key = (int64_t)mix64(column_seed + (uint64_t)row * GOLDEN); candidates[row].row = row; }
        qsort(candidates, rows, sizeof(candidate), compare_candidate);
        for (int i = 0; i < count; ++i) selected[i] = candidates[i].row;
        qsort(selected, count, sizeof(int), compare_int);
        portable_random r = stream(operand * 104729 + j); int has_diagonal = 0;
        for (int i = 0; i < count; ++i) {
            int row = selected[i]; if (triangular && ((lower && row < j) || (!lower && row > j))) continue;
            out.row_idx[out.nnz] = row;
            out.values[out.nnz++] = triangular && row == j ? 2.0 + fabs(random_double(&r)) : random_double(&r);
            if (row == j) has_diagonal = 1;
        }
        if (triangular && j < rows && !has_diagonal) { out.row_idx[out.nnz] = j; out.values[out.nnz++] = 2.0 + fabs(random_double(&r)); }
        /* Values must follow the row sort. Triangular fixture rows are already selected in ascending order,
           and the inserted diagonal can be out of order; move that final pair into place. */
        if (triangular && !has_diagonal) {
            int p = out.nnz - 1;
            while (p > out.col_ptr[j] && out.row_idx[p] < out.row_idx[p - 1]) {
                int ri = out.row_idx[p]; out.row_idx[p] = out.row_idx[p - 1]; out.row_idx[p - 1] = ri;
                double rv = out.values[p]; out.values[p] = out.values[p - 1]; out.values[p - 1] = rv; --p;
            }
        }
        out.col_ptr[j + 1] = out.nnz;
    }
    free(candidates); free(selected); return out;
}

static void free_sparse(sparse_fixture *a) { free(a->col_ptr); free(a->row_idx); free(a->values); memset(a, 0, sizeof(*a)); }
static const char *option(const bench_case *c, const char *name, const char *fallback) { for (int i = 0; i < c->option_count; ++i) if (!strcmp(c->option_names[i], name)) return c->option_values[i]; return fallback; }
static int flag(const bench_case *c, const char *name) { return !strcmp(option(c, name, "N"), "T"); }

static int expected_dimensions(const char *op) {
    static const char *one[] = { "dot","axpy","axpy-arithmetic","scal","nrm2","asum","sum","compensated-sum","iamax","swap","rot","rotm","rotmg","ssqd","dot4","axpy4","dot-axpy","symv","syr","syr2","trsv","trmv","spdot","spdot-raw","spdot-sparse","spaxpy","spaxpy-raw","spnrm2","spnrm2-indexed","spasum","spscatter","spscatter-raw","spgather","spgather-zero","spsymv","sptrsv","sptrmv","sparse-slices-scatter","sparse-slices-scatter-checked","sparse-slices-gather","sparse-slices-gather-clear","sparse-slices-clear","sparse-slices-clear-local","sparse-slices-reduce-dot-checked","sparse-slices-reduce-dot-local","sparse-slices-reduce-dot-unchecked","sparse-slices-max","sparse-slices-filter" };
    static const char *two[] = { "gemv","ger","symm","gemmt","syrk","syr2k","trsm","trmm","packed-trsm","pack-left","pack-right","pack-symmetric-left","pack-symmetric-right","pack-triangular-left","pack-triangular-right","write-left","write-right","clear-left-padding","clear-right-padding","spgemv","spsymm","sptrsm","sptrmm","spsyrk-dense","spsyrk-sparse","spadd" };
    static const char *three[] = { "gemm","gemm-tile","gemm-trsm","spmm","spgemm" };
    for (size_t i=0;i<sizeof(one)/sizeof(*one);++i) if(!strcmp(op,one[i])) return 1;
    for (size_t i=0;i<sizeof(two)/sizeof(*two);++i) if(!strcmp(op,two[i])) return 2;
    for (size_t i=0;i<sizeof(three)/sizeof(*three);++i) if(!strcmp(op,three[i])) return 3;
    return 0;
}

static int option_order(const char *name) { const char *names[] = {"density","mode","physical","side","uplo","transA","transB","diag"}; for(int i=0;i<8;++i) if(!strcmp(name,names[i])) return i; return -1; }
static int option_present(const bench_case *c,const char *name){for(int i=0;i<c->option_count;++i)if(!strcmp(c->option_names[i],name))return 1;return 0;}
static int operation_in(const char *operation,const char **values,size_t count){for(size_t i=0;i<count;++i)if(!strcmp(operation,values[i]))return 1;return 0;}
static int mode_operation(const char *op){return !strcmp(op,"spgemv")||!strcmp(op,"spmm")||!strcmp(op,"spgemm")||!strcmp(op,"spsymv")||!strcmp(op,"spsymm")||!strcmp(op,"sptrsv")||!strcmp(op,"sptrmv")||!strcmp(op,"sptrsm")||!strcmp(op,"sptrmm");}
static int oneshot_only_operation(const char *op){return !strcmp(op,"spsymv")||!strcmp(op,"spsymm")||!strcmp(op,"sptrsv")||!strcmp(op,"sptrmv")||!strcmp(op,"sptrsm")||!strcmp(op,"sptrmm");}
static int packed_operation(const char *op){const char *values[]={"gemm-tile","packed-trsm","gemm-trsm","pack-left","pack-right","pack-symmetric-left","pack-symmetric-right","pack-triangular-left","pack-triangular-right","write-left","write-right","clear-left-padding","clear-right-padding"};return operation_in(op,values,sizeof(values)/sizeof(*values));}
static int uplo_operation(const char *op){const char *values[]={"symv","syr","syr2","symm","gemmt","syrk","syr2k","trsv","trmv","trsm","trmm","packed-trsm","gemm-trsm","pack-symmetric-left","pack-symmetric-right","pack-triangular-left","pack-triangular-right","spsymv","spsymm","sptrsv","sptrmv","sptrsm","sptrmm","spsyrk-dense","spsyrk-sparse"};return operation_in(op,values,sizeof(values)/sizeof(*values));}
static int side_operation(const char *op){const char *values[]={"symm","trsm","trmm","spsymm","sptrsm","sptrmm"};return operation_in(op,values,sizeof(values)/sizeof(*values));}
static int transa_operation(const char *op){const char *values[]={"gemmt","syrk","syr2k","trsv","trmv","trsm","trmm","sptrsv","sptrmv","sptrsm","sptrmm"};return operation_in(op,values,sizeof(values)/sizeof(*values));}
static int diag_operation(const char *op){const char *values[]={"trsv","trmv","trsm","trmm","packed-trsm","gemm-trsm","pack-triangular-left","pack-triangular-right","sptrsv","sptrmv","sptrsm","sptrmm"};return operation_in(op,values,sizeof(values)/sizeof(*values));}
static int triangular_fixture_operation(const char *op){const char *values[]={"trsv","trmv","trsm","trmm","packed-trsm","gemm-trsm","pack-triangular-left","pack-triangular-right","spsymv","spsymm","sptrsv","sptrmv","sptrsm","sptrmm"};return operation_in(op,values,sizeof(values)/sizeof(*values));}
static int option_allowed(const char *op,const char *name,int sparse){
    if(!strcmp(name,"density"))return sparse;
    if(!strcmp(name,"mode"))return mode_operation(op);
    if(!strcmp(name,"physical"))return packed_operation(op);
    if(!strcmp(name,"side"))return side_operation(op);
    if(!strcmp(name,"uplo"))return uplo_operation(op);
    if(!strcmp(name,"transA"))return !strcmp(op,"gemv")||!strcmp(op,"gemm")||transa_operation(op);
    if(!strcmp(name,"transB"))return !strcmp(op,"gemm")||!strcmp(op,"gemmt");
    if(!strcmp(name,"diag"))return diag_operation(op);
    return 0;
}
static void copy_field(char *destination,size_t capacity,const char *source,int line_number,const char *name){size_t length=strlen(source);if(length>=capacity){fprintf(stderr,"line %d: %s is too long\n",line_number,name);exit(2);}memcpy(destination,source,length+1);}
static int valid_option(const char *name,const char *value){
    if(!strcmp(name,"density")){char *tail;double density=strtod(value,&tail);return !*tail&&density>0&&density<=1;}
    if(!strcmp(name,"mode"))return !strcmp(value,"prepared")||!strcmp(value,"oneshot");
    if(!strcmp(name,"physical"))return !strcmp(value,"4x4")||!strcmp(value,"8x4");
    if(!strcmp(name,"side"))return !strcmp(value,"L")||!strcmp(value,"R");
    if(!strcmp(name,"uplo"))return !strcmp(value,"L")||!strcmp(value,"U");
    if(!strcmp(name,"transA")||!strcmp(name,"transB"))return !strcmp(value,"N")||!strcmp(value,"T");
    if(!strcmp(name,"diag"))return !strcmp(value,"N")||!strcmp(value,"U");
    return 0;
}

static int parse_case(char *line, int line_number, bench_case *out) {
    while (*line == ' ' || *line == '\t') ++line;
    char *end = line + strlen(line); while (end > line && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t')) *--end = 0;
    if (!*line || *line == '#') return 0;
    if(*line=='+'||end[-1]=='+'||strstr(line,"++")){fprintf(stderr,"line %d: empty case field\n",line_number);exit(2);}
    copy_field(out->id,sizeof(out->id),line,line_number,"case");char copy[256];copy_field(copy,sizeof(copy),line,line_number,"case");
    char *save = NULL, *token = strtok_r(copy, "+", &save); if (!token) fail("missing operation"); copy_field(out->operation,sizeof(out->operation),token,line_number,"operation");
    int expected = expected_dimensions(token); if (!expected) { fprintf(stderr,"line %d: unknown operation %s\n",line_number,token); exit(2); }
    token = strtok_r(NULL, "+", &save); if (!token) fail("missing dimensions");
    if(*token=='x'||token[strlen(token)-1]=='x'||strstr(token,"xx")){fprintf(stderr,"line %d: empty dimension\n",line_number);exit(2);}char dimensions[64];copy_field(dimensions,sizeof(dimensions),token,line_number,"dimensions");char *dimension_save = NULL, *dimension = strtok_r(dimensions,"x",&dimension_save);
    while (dimension) { if (out->dim_count == MAX_DIMS || !*dimension) fail("malformed dimensions"); char *tail; long value = strtol(dimension,&tail,10); if(*tail || value<=0 || value>1000000) fail("invalid dimension"); out->dims[out->dim_count++] = (int)value; dimension=strtok_r(NULL,"x",&dimension_save); }
    if (out->dim_count != expected) { fprintf(stderr,"line %d: wrong dimension count\n",line_number); exit(2); }
    token = strtok_r(NULL, "+", &save); if (!token || (strcmp(token,"uniform") && strcmp(token,"triangular") && strcmp(token,"sparse-uniform") && strcmp(token,"sparse-triangular"))) fail("unknown fixture"); copy_field(out->fixture,sizeof(out->fixture),token,line_number,"fixture");
    int previous = -1;
    while ((token = strtok_r(NULL,"+",&save))) { char *equals = strchr(token,'='); if(!equals||equals==token||!equals[1]||strchr(equals+1,'=')) fail("malformed option"); *equals=0; int order=option_order(token); if(order<0 || order<=previous || out->option_count==MAX_OPTIONS||!valid_option(token,equals+1)) fail("unknown, invalid, duplicate, or noncanonical option"); previous=order;copy_field(out->option_names[out->option_count],sizeof(out->option_names[0]),token,line_number,"option name");copy_field(out->option_values[out->option_count],sizeof(out->option_values[0]),equals+1,line_number,"option value");++out->option_count; }
    int sparse = !strncmp(out->operation,"sp",2) || !strncmp(out->operation,"sparse-slices-",14);
    int triangular=triangular_fixture_operation(out->operation);const char *expected_fixture=sparse?(triangular?"sparse-triangular":"sparse-uniform"):(triangular?"triangular":"uniform");
    if(strcmp(out->fixture,expected_fixture))fail("incompatible fixture");
    for(int i=0;i<out->option_count;++i)if(!option_allowed(out->operation,out->option_names[i],sparse))fail("option is incompatible with operation");
    if(mode_operation(out->operation)!=option_present(out,"mode"))fail("missing or incompatible mode option");
    if(oneshot_only_operation(out->operation)&&strcmp(option(out,"mode",""),"oneshot"))fail("operation supports only mode=oneshot");
    if(sparse&&!option_present(out,"density"))fail("sparse cases require density");
    if(packed_operation(out->operation)&&!option_present(out,"physical"))fail("packed cases require physical shape");
    if(uplo_operation(out->operation)&&!option_present(out,"uplo"))fail("operation requires uplo");
    if(side_operation(out->operation)&&!option_present(out,"side"))fail("operation requires side");
    if(transa_operation(out->operation)&&!option_present(out,"transA"))fail("operation requires transA");
    if(!strcmp(out->operation,"gemmt")&&!option_present(out,"transB"))fail("gemmt requires transB");
    if(diag_operation(out->operation)&&!option_present(out,"diag"))fail("operation requires diag");
    const char *defaults[]={"side","L","uplo","L","transA","N","transB","N","diag","N"};
    for(int i=0;i<10;i+=2)if(option_present(out,defaults[i])&&!strcmp(option(out,defaults[i],""),defaults[i+1])){
        int required=!strcmp(defaults[i],"uplo")?uplo_operation(out->operation):!strcmp(defaults[i],"side")?side_operation(out->operation):!strcmp(defaults[i],"transA")?transa_operation(out->operation):!strcmp(defaults[i],"transB")?!strcmp(out->operation,"gemmt"):diag_operation(out->operation);
        if(!required)fail("redundant default option");
    }
    if(sparse&&option_present(out,"side")&&strcmp(option(out,"side",""),"L"))fail("sparse right-side cases are unsupported");
    if(option_present(out,"physical")){int physical_rows=0,physical_cols=0;if(sscanf(option(out,"physical",""),"%dx%d",&physical_rows,&physical_cols)!=2)fail("invalid physical shape");const char *op=out->operation;int rows_bounded=!strcmp(op,"gemm-tile")||!strcmp(op,"packed-trsm")||!strcmp(op,"gemm-trsm")||!strcmp(op,"pack-left")||!strcmp(op,"pack-symmetric-left")||!strcmp(op,"pack-triangular-left")||!strcmp(op,"write-left")||!strcmp(op,"clear-left-padding");int columns_bounded=!strcmp(op,"gemm-tile")||!strcmp(op,"packed-trsm")||!strcmp(op,"gemm-trsm")||!strcmp(op,"pack-right")||!strcmp(op,"pack-symmetric-right")||!strcmp(op,"pack-triangular-right")||!strcmp(op,"write-right")||!strcmp(op,"clear-right-padding");if(rows_bounded&&out->dims[0]>physical_rows)fail("logical rows exceed physical tile");if(columns_bounded&&out->dims[1]>physical_cols)fail("logical columns exceed physical tile");}
    return 1;
}

static bench_case *load_cases(const char *path, int *count) {
    FILE *file=fopen(path,"r"); if(!file){perror(path);exit(2);} size_t capacity=128; bench_case *cases=allocate(capacity,sizeof(*cases)); char *line=NULL; size_t length=0; int line_number=0;
    while(getline(&line,&length,file)>=0){++line_number; bench_case current={0}; if(!parse_case(line,line_number,&current))continue; for(int i=0;i<*count;++i)if(!strcmp(cases[i].id,current.id))fail("duplicate case"); if((size_t)*count==capacity){capacity*=2;cases=realloc(cases,capacity*sizeof(*cases));if(!cases)fail("out of memory");}cases[(*count)++]=current;}
    free(line);fclose(file);if(!*count)fail("case file contains no cases");return cases;
}

static uint64_t nanos(void) { struct timespec value; if(clock_gettime(CLOCK_MONOTONIC,&value))fail("clock_gettime failed"); return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec; }
static void copy_values(double *to,const double *from,int n){memcpy(to,from,(size_t)n*sizeof(double));}
static double consume(const double *values,int count){return count?values[0]+values[count-1]:0.0;}

static void fill_triangular(double *a,int n,int lower,int operand){fill_vector(a,n*n,operand);for(int j=0;j<n;++j)for(int i=0;i<n;++i){if((lower&&i<j)||(!lower&&i>j))a[i+j*n]=0.0;}for(int i=0;i<n;++i)a[i+i*n]=2.0+fabs(a[i+i*n]);}
static CBLAS_TRANSPOSE transpose_value(int transpose){return transpose?CblasTrans:CblasNoTrans;}
static CBLAS_UPLO uplo_value(int lower){return lower?CblasLower:CblasUpper;}
static CBLAS_SIDE side_value(int right){return right?CblasRight:CblasLeft;}
static CBLAS_DIAG diag_value(int unit){return unit?CblasUnit:CblasNonUnit;}
static void physical_shape(const bench_case *s,int *rows,int *cols){if(sscanf(option(s,"physical","4x4"),"%dx%d",rows,cols)!=2)fail("invalid physical shape");}
static void packed_left_fixture(double *logical,int rows,int depth,int physical_rows,int operand){double *packed=allocate(physical_rows*depth,sizeof(double));fill_vector(packed,physical_rows*depth,operand);for(int p=0;p<depth;++p)for(int i=0;i<rows;++i)logical[i+p*rows]=packed[i+p*physical_rows];free(packed);}
static void packed_right_fixture(double *logical,int depth,int cols,int physical_cols,int operand){double *packed=allocate(depth*physical_cols,sizeof(double));fill_vector(packed,depth*physical_cols,operand);for(int j=0;j<cols;++j)for(int p=0;p<depth;++p)logical[p+j*depth]=packed[j+p*physical_cols];free(packed);}
static void packed_output_fixture(double *logical,int rows,int cols,int physical_rows,int physical_cols,int operand){double *packed=allocate(physical_rows*physical_cols,sizeof(double));fill_vector(packed,physical_rows*physical_cols,operand);for(int j=0;j<cols;++j)for(int i=0;i<rows;++i)logical[i+j*rows]=packed[i+j*physical_rows];free(packed);}

static double invoke_dense(work *w){
    bench_case *s=w->spec;const char *op=s->operation;int *d=s->dims;int ta=flag(s,"transA"),tb=flag(s,"transB"),lower=!strcmp(option(s,"uplo","L"),"L"),right=!strcmp(option(s,"side","L"),"R"),unit=!strcmp(option(s,"diag","N"),"U");
    if(!strcmp(op,"dot"))return cblas_ddot(d[0],w->x,1,w->y,1);
    if(!strcmp(op,"axpy")||!strcmp(op,"axpy-arithmetic")){copy_values(w->y,w->initial,d[0]);cblas_daxpy(d[0],.875,w->x,1,w->y,1);return consume(w->y,d[0]);}
    if(!strcmp(op,"scal")){copy_values(w->x,w->initial,d[0]);cblas_dscal(d[0],.875,w->x,1);return consume(w->x,d[0]);}
    if(!strcmp(op,"nrm2"))return cblas_dnrm2(d[0],w->x,1);
    if(!strcmp(op,"asum"))return cblas_dasum(d[0],w->x,1);
    if(!strcmp(op,"iamax"))return (double)cblas_idamax(d[0],w->x,1);
#ifndef USE_MKL
    if(!strcmp(op,"sum"))return cblas_dsum(d[0],w->x,1);
#endif
    if(!strcmp(op,"swap")){copy_values(w->x,w->a,d[0]);copy_values(w->y,w->initial,d[0]);cblas_dswap(d[0],w->x,1,w->y,1);return w->x[0]+w->y[0];}
    if(!strcmp(op,"rot")){copy_values(w->x,w->a,d[0]);copy_values(w->y,w->initial,d[0]);cblas_drot(d[0],w->x,1,w->y,1,.8,.6);return w->x[0]+w->y[0];}
    if(!strcmp(op,"rotm")){copy_values(w->x,w->a,d[0]);copy_values(w->y,w->initial,d[0]);cblas_drotm(d[0],w->x,1,w->y,1,w->c);return w->x[0]+w->y[0];}
    if(!strcmp(op,"rotmg")){double d1=1,d2=1,x1=2,param[5];cblas_drotmg(&d1,&d2,&x1,1,param);return d1+d2+x1+param[0];}
    if(!strcmp(op,"dot4"))return cblas_ddot(d[0],w->a,1,w->x,1)+cblas_ddot(d[0],w->a+d[0],1,w->x,1)+cblas_ddot(d[0],w->a+2*d[0],1,w->x,1)+cblas_ddot(d[0],w->a+3*d[0],1,w->x,1);
    if(!strcmp(op,"axpy4")){copy_values(w->y,w->initial,d[0]);cblas_daxpy(d[0],.875,w->a,1,w->y,1);cblas_daxpy(d[0],-.875,w->a+d[0],1,w->y,1);cblas_daxpy(d[0],.875,w->a+2*d[0],1,w->y,1);cblas_daxpy(d[0],-.875,w->a+3*d[0],1,w->y,1);return consume(w->y,d[0]);}
    if(!strcmp(op,"dot-axpy")){copy_values(w->y,w->initial,d[0]);double result=cblas_ddot(d[0],w->x,1,w->a,1);cblas_daxpy(d[0],.875,w->x,1,w->y,1);return result+w->y[0];}
    if(!strcmp(op,"gemv")){int m=d[0],n=d[1];copy_values(w->y,w->initial,m);cblas_dgemv(CblasColMajor,transpose_value(ta),ta?n:m,ta?m:n,.875,w->a,ta?n:m,w->x,1,-.25,w->y,1);return consume(w->y,m);}
    if(!strcmp(op,"symv")){int n=d[0];copy_values(w->y,w->initial,n);cblas_dsymv(CblasColMajor,uplo_value(lower),n,.875,w->a,n,w->x,1,-.25,w->y,1);return consume(w->y,n);}
    if(!strcmp(op,"ger")){int m=d[0],n=d[1];copy_values(w->c,w->initial,m*n);cblas_dger(CblasColMajor,m,n,.875,w->x,1,w->y,1,w->c,m);return consume(w->c,m*n);}
    if(!strcmp(op,"syr")){int n=d[0];copy_values(w->c,w->initial,n*n);cblas_dsyr(CblasColMajor,uplo_value(lower),n,.875,w->x,1,w->c,n);return consume(w->c,n*n);}
    if(!strcmp(op,"syr2")){int n=d[0];copy_values(w->c,w->initial,n*n);cblas_dsyr2(CblasColMajor,uplo_value(lower),n,.875,w->x,1,w->y,1,w->c,n);return consume(w->c,n*n);}
    if(!strcmp(op,"trsv")||!strcmp(op,"trmv")){int n=d[0];copy_values(w->x,w->initial,n);if(op[2]=='s')cblas_dtrsv(CblasColMajor,uplo_value(lower),transpose_value(ta),diag_value(unit),n,w->a,n,w->x,1);else cblas_dtrmv(CblasColMajor,uplo_value(lower),transpose_value(ta),diag_value(unit),n,w->a,n,w->x,1);return consume(w->x,n);}
    if(!strcmp(op,"gemm")){int m=d[0],n=d[1],k=d[2],ar=ta?k:m,br=tb?n:k;copy_values(w->c,w->initial,m*n);cblas_dgemm(CblasColMajor,transpose_value(ta),transpose_value(tb),m,n,k,.875,w->a,ar,w->b,br,-.25,w->c,m);return consume(w->c,m*n);}
    if(!strcmp(op,"symm")){int m=d[0],n=d[1],order=right?n:m;copy_values(w->c,w->initial,m*n);cblas_dsymm(CblasColMajor,side_value(right),uplo_value(lower),m,n,.875,w->a,order,w->b,m,-.25,w->c,m);return consume(w->c,m*n);}
    if(!strcmp(op,"gemmt")){int n=d[0],k=d[1],ar=ta?k:n,br=tb?n:k;copy_values(w->c,w->initial,n*n);cblas_dgemmt(CblasColMajor,uplo_value(lower),transpose_value(ta),transpose_value(tb),n,k,.875,w->a,ar,w->b,br,-.25,w->c,n);return consume(w->c,n*n);}
    if(!strcmp(op,"syrk")){int n=d[0],k=d[1],ar=ta?k:n;copy_values(w->c,w->initial,n*n);cblas_dsyrk(CblasColMajor,uplo_value(lower),transpose_value(ta),n,k,.875,w->a,ar,-.25,w->c,n);return consume(w->c,n*n);}
    if(!strcmp(op,"syr2k")){int n=d[0],k=d[1],ar=ta?k:n;copy_values(w->c,w->initial,n*n);cblas_dsyr2k(CblasColMajor,uplo_value(lower),transpose_value(ta),n,k,.875,w->a,ar,w->b,ar,-.25,w->c,n);return consume(w->c,n*n);}
    if(!strcmp(op,"trsm")||!strcmp(op,"trmm")){int m=d[0],n=d[1],order=right?n:m;copy_values(w->b,w->initial,m*n);if(op[2]=='s')cblas_dtrsm(CblasColMajor,side_value(right),uplo_value(lower),transpose_value(ta),diag_value(unit),m,n,.875,w->a,order,w->b,m);else cblas_dtrmm(CblasColMajor,side_value(right),uplo_value(lower),transpose_value(ta),diag_value(unit),m,n,.875,w->a,order,w->b,m);return consume(w->b,m*n);}
    if(!strcmp(op,"gemm-tile")){int m=d[0],n=d[1],k=d[2];copy_values(w->c,w->initial,m*n);cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,w->a,m,w->b,k,1,w->c,m);return consume(w->c,m*n);}
    if(!strcmp(op,"packed-trsm")){int m=d[0],n=d[1];copy_values(w->b,w->initial,m*n);cblas_dtrsm(CblasColMajor,CblasRight,uplo_value(lower),CblasNoTrans,diag_value(unit),m,n,1,w->a,n,w->b,m);return consume(w->b,m*n);}
    if(!strcmp(op,"gemm-trsm")){int m=d[0],n=d[1],k=d[2];copy_values(w->c,w->initial,m*n);cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,-1,w->a,m,w->b,k,1,w->c,m);cblas_dtrsm(CblasColMajor,CblasRight,uplo_value(lower),CblasNoTrans,diag_value(unit),m,n,1,w->x,n,w->c,m);return consume(w->c,m*n);}
    return 0;
}

static int is_layout(const char *op){return !strncmp(op,"pack-",5)||!strncmp(op,"write-",6)||!strncmp(op,"clear-",6);}

static void setup_dense(work *w){
    bench_case *s=w->spec;const char *op=s->operation;int *d=s->dims;int ta=flag(s,"transA"),tb=flag(s,"transB"),lower=!strcmp(option(s,"uplo","L"),"L"),right=!strcmp(option(s,"side","L"),"R");
    w->supported=1;w->comparison="direct";w->timing="reset-and-arithmetic";w->invoke=invoke_dense;
    if(is_layout(op)||!strcmp(op,"ssqd")||!strcmp(op,"compensated-sum")){w->supported=0;w->comparison="unsupported";w->timing=is_layout(op)?"layout":"arithmetic";return;}
#ifdef USE_MKL
    if(!strcmp(op,"sum")){w->supported=0;w->comparison="unsupported";w->timing="arithmetic";return;}
#endif
    if(!strcmp(op,"dot")||!strcmp(op,"nrm2")||!strcmp(op,"asum")||!strcmp(op,"sum")||!strcmp(op,"iamax")){w->x=allocate(d[0],sizeof(double));w->y=allocate(d[0],sizeof(double));fill_vector(w->x,d[0],1);fill_vector(w->y,d[0],2);w->timing="arithmetic";return;}
    if(!strcmp(op,"axpy")||!strcmp(op,"axpy-arithmetic")||!strcmp(op,"scal")){w->x=allocate(d[0],sizeof(double));w->y=allocate(d[0],sizeof(double));w->initial=allocate(d[0],sizeof(double));fill_vector(w->x,d[0],1);fill_vector(w->initial,d[0],!strcmp(op,"scal")?1:2);if(!strcmp(op,"scal"))copy_values(w->initial,w->x,d[0]);return;}
    if(!strcmp(op,"swap")||!strcmp(op,"rot")||!strcmp(op,"rotm")){w->a=allocate(d[0],sizeof(double));w->x=allocate(d[0],sizeof(double));w->y=allocate(d[0],sizeof(double));w->initial=allocate(d[0],sizeof(double));fill_vector(w->a,d[0],1);fill_vector(w->initial,d[0],2);if(!strcmp(op,"rotm")){w->c=allocate(5,sizeof(double));w->c[0]=-1;w->c[1]=1;w->c[2]=-.5;w->c[3]=.5;w->c[4]=1;}return;}
    if(!strcmp(op,"rotmg")){w->timing="arithmetic";return;}
    if(!strcmp(op,"dot4")||!strcmp(op,"axpy4")||!strcmp(op,"dot-axpy")){w->a=allocate(4*d[0],sizeof(double));w->x=allocate(d[0],sizeof(double));w->y=allocate(d[0],sizeof(double));w->initial=allocate(d[0],sizeof(double));fill_vector(w->a,4*d[0],!strcmp(op,"axpy4")?1:2);fill_vector(w->x,d[0],1);fill_vector(w->initial,d[0],!strcmp(op,"axpy4")?2:3);w->comparison="composed";if(!strcmp(op,"dot4"))w->timing="arithmetic";return;}
    if(!strcmp(op,"gemv")){int m=d[0],n=d[1],ar=ta?n:m,ac=ta?m:n;w->a=allocate(ar*ac,sizeof(double));w->x=allocate(n,sizeof(double));w->y=allocate(m,sizeof(double));w->initial=allocate(m,sizeof(double));fill_vector(w->a,ar*ac,1);fill_vector(w->x,n,2);fill_vector(w->initial,m,3);return;}
    if(!strcmp(op,"symv")||!strcmp(op,"syr")||!strcmp(op,"syr2")||!strcmp(op,"trsv")||!strcmp(op,"trmv")){int n=d[0];w->a=allocate(n*n,sizeof(double));w->c=allocate(n*n,sizeof(double));w->initial=allocate(n*n>n?n*n:n,sizeof(double));w->x=allocate(n,sizeof(double));w->y=allocate(n,sizeof(double));fill_vector(w->a,n*n,1);fill_vector(w->initial,n*n>n?n*n:n,!strcmp(op,"symv")?3:(!strncmp(op,"tr",2)?2:1));fill_vector(w->x,n,2);fill_vector(w->y,n,3);if(op[0]=='t')fill_triangular(w->a,n,lower,1);return;}
    if(!strcmp(op,"ger")){int m=d[0],n=d[1];w->c=allocate(m*n,sizeof(double));w->initial=allocate(m*n,sizeof(double));w->x=allocate(m,sizeof(double));w->y=allocate(n,sizeof(double));fill_vector(w->initial,m*n,1);fill_vector(w->x,m,2);fill_vector(w->y,n,3);return;}
    if(!strcmp(op,"gemm")){int m=d[0],n=d[1],k=d[2],as=(ta?k:m)*(ta?m:k),bs=(tb?n:k)*(tb?k:n);w->a=allocate(as,sizeof(double));w->b=allocate(bs,sizeof(double));w->c=allocate(m*n,sizeof(double));w->initial=allocate(m*n,sizeof(double));fill_vector(w->a,as,1);fill_vector(w->b,bs,2);fill_vector(w->initial,m*n,3);return;}
    if(!strcmp(op,"symm")){int m=d[0],n=d[1],order=right?n:m;w->a=allocate(order*order,sizeof(double));w->b=allocate(m*n,sizeof(double));w->c=allocate(m*n,sizeof(double));w->initial=allocate(m*n,sizeof(double));fill_vector(w->a,order*order,1);fill_vector(w->b,m*n,2);fill_vector(w->initial,m*n,3);return;}
    if(!strcmp(op,"gemmt")||!strcmp(op,"syrk")||!strcmp(op,"syr2k")){int n=d[0],k=d[1],as=(ta?k:n)*(ta?n:k);w->a=allocate(as,sizeof(double));w->b=allocate((!strcmp(op,"gemmt")?((tb?n:k)*(tb?k:n)):as),sizeof(double));w->c=allocate(n*n,sizeof(double));w->initial=allocate(n*n,sizeof(double));fill_vector(w->a,as,1);fill_vector(w->b,!strcmp(op,"gemmt")?((tb?n:k)*(tb?k:n)):as,2);fill_vector(w->initial,n*n,3);return;}
    if(!strcmp(op,"trsm")||!strcmp(op,"trmm")){int m=d[0],n=d[1],order=right?n:m;w->a=allocate(order*order,sizeof(double));w->b=allocate(m*n,sizeof(double));w->initial=allocate(m*n,sizeof(double));fill_triangular(w->a,order,lower,1);fill_vector(w->initial,m*n,2);return;}
    if(!strcmp(op,"gemm-tile")){int m=d[0],n=d[1],k=d[2],pr,pc;physical_shape(s,&pr,&pc);w->a=allocate(m*k,sizeof(double));w->b=allocate(k*n,sizeof(double));w->c=allocate(m*n,sizeof(double));w->initial=allocate(m*n,sizeof(double));packed_left_fixture(w->a,m,k,pr,1);packed_right_fixture(w->b,k,n,pc,2);packed_output_fixture(w->initial,m,n,pr,pc,4);w->comparison="partial";w->timing="arithmetic-only";return;}
    if(!strcmp(op,"packed-trsm")){int m=d[0],n=d[1],pr,pc;physical_shape(s,&pr,&pc);w->a=allocate(n*n,sizeof(double));w->b=allocate(m*n,sizeof(double));w->initial=allocate(m*n,sizeof(double));fill_triangular(w->a,n,lower,20);packed_output_fixture(w->initial,m,n,pr,pc,4);w->comparison="partial";w->timing="arithmetic-only";return;}
    if(!strcmp(op,"gemm-trsm")){int m=d[0],n=d[1],k=d[2],pr,pc;physical_shape(s,&pr,&pc);w->a=allocate(m*k,sizeof(double));w->b=allocate(k*n,sizeof(double));w->c=allocate(m*n,sizeof(double));w->initial=allocate(m*n,sizeof(double));w->x=allocate(n*n,sizeof(double));packed_left_fixture(w->a,m,k,pr,1);packed_right_fixture(w->b,k,n,pc,2);packed_output_fixture(w->initial,m,n,pr,pc,4);fill_triangular(w->x,n,lower,20);w->comparison="composed";w->timing="arithmetic-only";return;}
    w->supported=0;w->comparison="unsupported";
}

#ifdef USE_MKL
enum { SPARSE_NON_TRANSPOSE=10,SPARSE_TRANSPOSE=11,TYPE_GENERAL=20,TYPE_SYMMETRIC=21,TYPE_TRIANGULAR=23,FILL_LOWER=40,FILL_UPPER=41,DIAG_NON_UNIT=50,DIAG_UNIT=51,COLUMN_MAJOR=102 };

static sparse_fixture to_csr(const sparse_fixture *csc){
    sparse_fixture csr={csc->rows,csc->cols,csc->nnz,allocate((size_t)csc->rows+1,sizeof(int)),allocate(csc->nnz,sizeof(int)),allocate(csc->nnz,sizeof(double))};
    for(int p=0;p<csc->nnz;++p)csr.col_ptr[csc->row_idx[p]+1]++;
    for(int i=0;i<csc->rows;++i)csr.col_ptr[i+1]+=csr.col_ptr[i];
    int *cursor=allocate(csc->rows,sizeof(int));memcpy(cursor,csr.col_ptr,(size_t)csc->rows*sizeof(int));
    for(int j=0;j<csc->cols;++j)for(int p=csc->col_ptr[j];p<csc->col_ptr[j+1];++p){int q=cursor[csc->row_idx[p]]++;csr.row_idx[q]=j;csr.values[q]=csc->values[p];}
    free(cursor);return csr;
}

static sparse_matrix_t create_handle(sparse_fixture *csr){
    sparse_matrix_t handle=NULL;int status=mkl_sparse_d_create_csr(&handle,0,csr->rows,csr->cols,csr->col_ptr,csr->col_ptr+1,csr->row_idx,csr->values);
    if(status)fail("mkl_sparse_d_create_csr failed");
    if(mkl_sparse_optimize(handle))fail("mkl_sparse_optimize failed");
    return handle;
}

static matrix_descr descriptor_for(const bench_case *s){
    matrix_descr descriptor={TYPE_GENERAL,FILL_LOWER,DIAG_NON_UNIT};
    if(!strcmp(s->operation,"spsymv")||!strcmp(s->operation,"spsymm"))descriptor.type=TYPE_SYMMETRIC;
    if(!strncmp(s->operation,"sptr",4))descriptor.type=TYPE_TRIANGULAR;
    descriptor.mode=!strcmp(option(s,"uplo","L"),"L")?FILL_LOWER:FILL_UPPER;
    descriptor.diag=!strcmp(option(s,"diag","N"),"U")?DIAG_UNIT:DIAG_NON_UNIT;
    return descriptor;
}

static sparse_matrix_t operation_handle(work *w,int which,int *owned,sparse_fixture *temporary){
    sparse_matrix_t prepared=which?w->handle_b:w->handle_a;if(prepared){*owned=0;return prepared;}
    *owned=1;*temporary=to_csr(which?&w->sb:&w->sa);return create_handle(temporary);
}

static double invoke_sparse(work *w){
    bench_case *s=w->spec;const char *op=s->operation;int *d=s->dims;matrix_descr descriptor=descriptor_for(s);int operation=flag(s,"transA")?SPARSE_TRANSPOSE:SPARSE_NON_TRANSPOSE;
    if(!strcmp(op,"spdot"))return cblas_ddoti(w->sa.nnz,w->sa.values,w->sa.row_idx,w->y);
    if(!strcmp(op,"spaxpy")){copy_values(w->y,w->initial,d[0]);cblas_daxpyi(w->sa.nnz,.875,w->sa.values,w->sa.row_idx,w->y);return consume(w->y,d[0]);}
    if(!strcmp(op,"spscatter")){copy_values(w->y,w->initial,d[0]);cblas_dsctr(w->sa.nnz,w->sa.values,w->sa.row_idx,w->y);return consume(w->y,d[0]);}
    if(!strcmp(op,"spgather")){copy_values(w->y,w->initial,d[0]);cblas_dgthr(w->sa.nnz,w->y,w->x,w->sa.row_idx);return consume(w->x,w->sa.nnz);}
    if(!strcmp(op,"spgather-zero")){copy_values(w->y,w->initial,d[0]);cblas_dgthrz(w->sa.nnz,w->y,w->x,w->sa.row_idx);return consume(w->x,w->sa.nnz)+consume(w->y,d[0]);}
    int own_a=0,own_b=0;sparse_fixture temporary_a={0},temporary_b={0};sparse_matrix_t a=operation_handle(w,0,&own_a,&temporary_a),b=NULL,result=NULL;double answer=0;
    if(!strcmp(op,"spgemv")||!strcmp(op,"spsymv")){copy_values(w->y,w->initial,d[0]);if(mkl_sparse_d_mv(operation,.875,a,descriptor,w->x,-.25,w->y))fail("mkl_sparse_d_mv failed");answer=consume(w->y,d[0]);}
    else if(!strcmp(op,"spsymm")){int n=d[0],rhs=d[1];copy_values(w->c,w->initial,n*rhs);if(mkl_sparse_d_mm(operation,.875,a,descriptor,COLUMN_MAJOR,w->b,rhs,n,-.25,w->c,n))fail("mkl_sparse_d_mm symmetric failed");answer=consume(w->c,n*rhs);}
    else if(!strcmp(op,"spmm")){int m=d[0],n=d[1],k=d[2];copy_values(w->c,w->initial,m*n);if(mkl_sparse_d_mm(operation,.875,a,descriptor,COLUMN_MAJOR,w->b,n,k,-.25,w->c,m))fail("mkl_sparse_d_mm failed");answer=consume(w->c,m*n);}
    else if(!strcmp(op,"spgemm")){b=operation_handle(w,1,&own_b,&temporary_b);if(mkl_sparse_spmm(SPARSE_NON_TRANSPOSE,a,b,&result))fail("mkl_sparse_spmm failed");answer=(double)(uintptr_t)result;if(mkl_sparse_destroy(result))fail("mkl_sparse_destroy result failed");}
    else if(!strcmp(op,"sptrsv")){copy_values(w->y,w->initial,d[0]);if(mkl_sparse_d_trsv(operation,1,a,descriptor,w->x,w->y))fail("mkl_sparse_d_trsv failed");answer=consume(w->y,d[0]);}
    else if(!strcmp(op,"sptrmv")){copy_values(w->y,w->initial,d[0]);if(mkl_sparse_d_mv(operation,1,a,descriptor,w->x,0,w->y))fail("mkl_sparse_d_mv triangular failed");answer=consume(w->y,d[0]);}
    else if(!strcmp(op,"sptrsm")){int order=d[0],rhs=d[1];copy_values(w->c,w->initial,order*rhs);if(mkl_sparse_d_trsm(operation,.875,a,descriptor,COLUMN_MAJOR,w->b,rhs,order,w->c,order))fail("mkl_sparse_d_trsm failed");answer=consume(w->c,order*rhs);}
    else if(!strcmp(op,"sptrmm")){int order=d[0],rhs=d[1];copy_values(w->c,w->initial,order*rhs);if(mkl_sparse_d_mm(operation,.875,a,descriptor,COLUMN_MAJOR,w->b,rhs,order,0,w->c,order))fail("mkl_sparse_d_mm triangular failed");answer=consume(w->c,order*rhs);}
    else if(!strcmp(op,"spsyrk-dense")){int n=d[0];copy_values(w->c,w->initial,n*n);if(mkl_sparse_d_syrkd(SPARSE_NON_TRANSPOSE,a,.875,-.25,w->c,COLUMN_MAJOR,n))fail("mkl_sparse_d_syrkd failed");answer=consume(w->c,n*n);}
    else if(!strcmp(op,"spsyrk-sparse")){if(mkl_sparse_syrk(SPARSE_NON_TRANSPOSE,a,&result))fail("mkl_sparse_syrk failed");answer=(double)(uintptr_t)result;if(mkl_sparse_destroy(result))fail("mkl_sparse_destroy result failed");}
    else if(!strcmp(op,"spadd")){b=operation_handle(w,1,&own_b,&temporary_b);if(mkl_sparse_d_add(SPARSE_NON_TRANSPOSE,a,.875,b,&result))fail("mkl_sparse_d_add failed");answer=(double)(uintptr_t)result;if(mkl_sparse_destroy(result))fail("mkl_sparse_destroy result failed");}
    if(own_b&&mkl_sparse_destroy(b))fail("mkl_sparse_destroy B failed");
    if(own_a&&mkl_sparse_destroy(a))fail("mkl_sparse_destroy A failed");
    free_sparse(&temporary_b);free_sparse(&temporary_a);
    return answer;
}

static void setup_sparse(work *w){
    bench_case *s=w->spec;int *d=s->dims;double density=strtod(option(s,"density","0.01"),NULL);int lower=!strcmp(option(s,"uplo","L"),"L");
    w->supported=1;w->comparison="direct";w->timing=!strcmp(option(s,"mode","oneshot"),"prepared")?"prepared":"oneshot";w->invoke=invoke_sparse;
    if(!strncmp(s->operation,"sparse-slices-",14)){w->supported=0;w->comparison="unsupported";w->timing="sparse-slices";return;}
    if(!strcmp(s->operation,"spdot-raw")||!strcmp(s->operation,"spdot-sparse")||
       !strcmp(s->operation,"spaxpy-raw")||!strcmp(s->operation,"spnrm2")||
       !strcmp(s->operation,"spnrm2-indexed")||!strcmp(s->operation,"spasum")||
       !strcmp(s->operation,"spscatter-raw")){
        w->supported=0;w->comparison="unsupported";w->timing="arithmetic";return;
    }
    if(!strcmp(s->operation,"spdot")||!strcmp(s->operation,"spaxpy")||!strcmp(s->operation,"spscatter")||!strcmp(s->operation,"spgather")||!strcmp(s->operation,"spgather-zero")){w->sa=make_sparse(d[0],1,density,1,0,1);w->y=allocate(d[0],sizeof(double));w->initial=allocate(d[0],sizeof(double));w->x=allocate(w->sa.nnz,sizeof(double));fill_vector(w->initial,d[0],2);copy_values(w->y,w->initial,d[0]);w->timing=!strcmp(s->operation,"spdot")?"arithmetic":"reset-and-arithmetic";return;}
    int rows=d[0],cols=1,triangular=!strncmp(s->operation,"sptr",4)||!strcmp(s->operation,"spsymv")||!strcmp(s->operation,"spsymm");
    if(!strcmp(s->operation,"spgemv"))cols=d[1];
    else if(!strcmp(s->operation,"spmm")||!strcmp(s->operation,"spgemm"))cols=d[2];
    else if(!strcmp(s->operation,"spsyrk-dense")||!strcmp(s->operation,"spsyrk-sparse")||!strcmp(s->operation,"spadd"))cols=d[1];
    else cols=rows;
    w->sa=make_sparse(rows,cols,density,1,triangular,lower);
    if(!strcmp(s->operation,"spgemm"))w->sb=make_sparse(cols,d[1],density,2,0,1);
    if(!strcmp(s->operation,"spadd"))w->sb=make_sparse(rows,cols,density,2,0,1);
    if(!strcmp(option(s,"mode","oneshot"),"prepared")){w->csr_a=to_csr(&w->sa);w->handle_a=create_handle(&w->csr_a);}
    if(!strcmp(s->operation,"spgemv")||!strcmp(s->operation,"spsymv")||!strcmp(s->operation,"sptrsv")||!strcmp(s->operation,"sptrmv")){w->x=allocate(cols,sizeof(double));w->y=allocate(rows,sizeof(double));w->initial=allocate(rows,sizeof(double));fill_vector(w->x,cols,2);fill_vector(w->initial,rows,3);}
    if(!strcmp(s->operation,"spmm")){int n=d[1],k=d[2];w->b=allocate(k*n,sizeof(double));w->c=allocate(rows*n,sizeof(double));w->initial=allocate(rows*n,sizeof(double));fill_vector(w->b,k*n,2);fill_vector(w->initial,rows*n,3);}
    if(!strcmp(s->operation,"spsymm")){int rhs=d[1];w->b=allocate(rows*rhs,sizeof(double));w->c=allocate(rows*rhs,sizeof(double));w->initial=allocate(rows*rhs,sizeof(double));fill_vector(w->b,rows*rhs,2);fill_vector(w->initial,rows*rhs,3);}
    if(!strcmp(s->operation,"sptrsm")||!strcmp(s->operation,"sptrmm")){int rhs=d[1];w->b=allocate(rows*rhs,sizeof(double));w->c=allocate(rows*rhs,sizeof(double));w->initial=allocate(rows*rhs,sizeof(double));fill_vector(w->b,rows*rhs,2);fill_vector(w->initial,rows*rhs,2);}
    if(!strcmp(s->operation,"spsyrk-dense")){w->c=allocate(rows*rows,sizeof(double));w->initial=allocate(rows*rows,sizeof(double));fill_vector(w->initial,rows*rows,2);}
    if(!strcmp(s->operation,"spgemm")||!strcmp(s->operation,"spsyrk-sparse")||!strcmp(s->operation,"spadd"))w->comparison="partial";
}
#endif

static void setup_work(work *w,bench_case *spec){memset(w,0,sizeof(*w));w->spec=spec;if(!strncmp(spec->operation,"sp",2)||!strncmp(spec->operation,"sparse-slices-",14)){
#ifdef USE_MKL
setup_sparse(w);
#else
w->supported=0;w->comparison="unsupported";w->timing=!strncmp(spec->operation,"sparse-slices-",14)?"sparse-slices":option(spec,"mode","arithmetic");
#endif
}else setup_dense(w);}

static void free_work(work *w){
#ifdef USE_MKL
if(w->handle_b)mkl_sparse_destroy(w->handle_b);
if(w->handle_a)mkl_sparse_destroy(w->handle_a);
free_sparse(&w->csr_a);free_sparse(&w->csr_b);
#endif
free_sparse(&w->sa);free_sparse(&w->sb);free(w->a);free(w->b);free(w->c);free(w->initial);free(w->x);free(w->y);free(w->indices);
}

static void fixture_check(void){
    double values[16];fill_vector(values,16,1);if(digest(values,16)!=UINT64_C(0x173cc3a80546954d))fail("dense fixture golden mismatch");
    sparse_fixture sparse=make_sparse(17,5,.2,2,0,1);if(digest(sparse.values,sparse.nnz)!=UINT64_C(0xe6ac2de9cae9ebe8))fail("sparse fixture golden mismatch");
    int expected[]={0,3,6,9,12,15};for(int i=0;i<6;++i)if(sparse.col_ptr[i]!=expected[i])fail("sparse support golden mismatch");free_sparse(&sparse);
}

static void numerical_check(void){
    double x[19],y[19];fill_vector(x,19,1);fill_vector(y,19,2);double reference=0;for(int i=0;i<19;++i)reference+=x[i]*y[i];double actual=cblas_ddot(19,x,1,y,1);if(fabs(actual-reference)>1e-12*(1+fabs(reference)))fail("dense dot numerical check failed");
    int m=7,n=5,k=9;double *a=allocate(m*k,sizeof(double)),*b=allocate(k*n,sizeof(double)),*c=allocate(m*n,sizeof(double)),*oracle=allocate(m*n,sizeof(double));fill_vector(a,m*k,1);fill_vector(b,k*n,2);fill_vector(c,m*n,3);copy_values(oracle,c,m*n);cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,.875,a,m,b,k,-.25,c,m);for(int j=0;j<n;++j)for(int i=0;i<m;++i){double sum=0;for(int p=0;p<k;++p)sum+=a[i+p*m]*b[p+j*k];oracle[i+j*m]=.875*sum-.25*oracle[i+j*m];}for(int i=0;i<m*n;++i)if(fabs(c[i]-oracle[i])>2e-12*(1+fabs(oracle[i])))fail("dense gemm numerical check failed");free(a);free(b);free(c);free(oracle);
#ifdef USE_MKL
    sparse_fixture sa=make_sparse(11,7,.3,4,0,1),csr=to_csr(&sa);sparse_matrix_t handle=create_handle(&csr);double sx[7],sy[11]={0},expected[11]={0};fill_vector(sx,7,5);matrix_descr descriptor={TYPE_GENERAL,FILL_LOWER,DIAG_NON_UNIT};if(mkl_sparse_d_mv(SPARSE_NON_TRANSPOSE,1,handle,descriptor,sx,0,sy))fail("sparse numerical call failed");for(int j=0;j<7;++j)for(int p=sa.col_ptr[j];p<sa.col_ptr[j+1];++p)expected[sa.row_idx[p]]+=sa.values[p]*sx[j];for(int i=0;i<11;++i)if(fabs(sy[i]-expected[i])>2e-12*(1+fabs(expected[i])))fail("sparse gemv numerical check failed");mkl_sparse_destroy(handle);free_sparse(&sa);free_sparse(&csr);
#endif
}

static const char *argument_value(int argc,char **argv,const char *name,const char *fallback){size_t length=strlen(name);for(int i=1;i<argc;++i)if(!strncmp(argv[i],name,length)&&argv[i][length]=='=')return argv[i]+length+1;return fallback;}

int main(int argc,char **argv){
    const char *cases_path=argument_value(argc,argv,"--cases","koblas-bench/cases.txt");const char *output_path=argument_value(argc,argv,"--output",NULL);if(!output_path)fail("--output is required");
    int warmups=atoi(argument_value(argc,argv,"--warmups","3")),samples=atoi(argument_value(argc,argv,"--samples","5"));long target_ms=strtol(argument_value(argc,argv,"--target-ms","100"),NULL,10);if(warmups<0||samples<1||target_ms<1)fail("invalid timing settings");uint64_t target_ns=(uint64_t)target_ms*UINT64_C(1000000);
    const char *pass=argument_value(argc,argv,"--pass","1"),*commit=argument_value(argc,argv,"--source-commit","unknown"),*dirty=argument_value(argc,argv,"--dirty","unknown");
#ifdef USE_MKL
    MKL_Set_Num_Threads(1);MKL_Set_Dynamic(0);char runtime[256]={0};MKL_Get_Version_String(runtime,sizeof(runtime));const char *implementation="onemkl";
#else
    openblas_set_num_threads(1);char runtime[256]={0};snprintf(runtime,sizeof(runtime),"%s",openblas_get_config());const char *implementation="openblas";
#endif
    size_t runtime_length=strlen(runtime);snprintf(runtime+runtime_length,sizeof(runtime)-runtime_length,"; compiler=%s",__VERSION__);
    for(char *p=runtime;*p;++p)if(*p==','||*p=='\n'||*p=='\r')*p=';';
    fixture_check();numerical_check();
    int case_count=0;bench_case *cases=load_cases(cases_path,&case_count);
    FILE *output=fopen(output_path,"w");if(!output){perror(output_path);exit(2);}fprintf(output,"schema,case,implementation,workload_version,fixture_version,pass,sample,operations,elapsed_ns,ns_per_op,unit,status,comparison_kind,timing_mode,source_commit,dirty,runtime,threads,warmups,target_ns\n");
    volatile double sink=0;
    for(int index=0;index<case_count;++index){work w;setup_work(&w,&cases[index]);if(!w.supported){fprintf(output,"3,%s,%s," WORKLOAD_VERSION "," FIXTURE_VERSION ",%s,0,0,0,,ns,unsupported,%s,%s,%s,%s,%s,1,%d,%" PRIu64 "\n",cases[index].id,implementation,pass,w.comparison,w.timing,commit,dirty,runtime,warmups,target_ns);free_work(&w);continue;}
        for(int i=0;i<warmups;++i)sink+=w.invoke(&w);
        int operations=1;while(operations<1000000){uint64_t start=nanos();for(int i=0;i<operations;++i)sink+=w.invoke(&w);uint64_t elapsed=nanos()-start;if(elapsed>=(uint64_t)target_ms*250000)break;operations*=2;}
        for(int sample=1;sample<=samples;++sample){uint64_t start=nanos();for(int i=0;i<operations;++i)sink+=w.invoke(&w);uint64_t elapsed=nanos()-start;fprintf(output,"3,%s,%s," WORKLOAD_VERSION "," FIXTURE_VERSION ",%s,%d,%d,%" PRIu64 ",%.17g,ns,ok,%s,%s,%s,%s,%s,1,%d,%" PRIu64 "\n",cases[index].id,implementation,pass,sample,operations,elapsed,(double)elapsed/operations,w.comparison,w.timing,commit,dirty,runtime,warmups,target_ns);}
        free_work(&w);
    }
    fclose(output);free(cases);fprintf(stderr,"wrote %d cases to %s (%s, sink=%g)\n",case_count,output_path,implementation,(double)sink);return 0;
}
