#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#define KOBLAS_KERNELS_IMPLEMENTATION
#include "koblas/src/nativeInterop/cinterop/koblas_kernels.h"
#define CLONES __attribute__((target_clones("avx2", "default")))
typedef void (*scale_fn)(int,double,double*,int);
static uint64_t now(clockid_t clock){struct timespec t;if(clock_gettime(clock,&t))abort();return(uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static void baseline(int n,double a,double*v,int stride){(void)stride;koblas_dense_scale(v,0,a,n);}
CLONES static void aligned(int n,double a,double*v,int stride){
 (void)stride;if(a==1||n<=0)return;
 while(n>0&&((uintptr_t)v&31)){*v++*=a;--n;}if(!n)return;
 v=__builtin_assume_aligned(v,32);for(int i=0;i<n;++i)v[i]*=a;
}
#define STEP(q) koblas_v4d x##q; __builtin_memcpy(&x##q,v+i+q*4,32); x##q*=multiplier;
#define STORE(q) __builtin_memcpy(v+i+q*4,&x##q,32);
#define EIGHT(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
CLONES static void unrolled(int n,double a,double*v,int stride){
 (void)stride;if(a==1)return;int i=0;koblas_v4d multiplier={a,a,a,a};
 for(;i<=n-32;i+=32){EIGHT(STEP) EIGHT(STORE)}for(;i<n;++i)v[i]*=a;
}
CLONES static void aligned_unrolled(int n,double a,double*v,int stride){
 (void)stride;if(a==1||n<=0)return;
 while(n>0&&((uintptr_t)v&31)){*v++*=a;--n;}if(!n)return;v=__builtin_assume_aligned(v,32);
 int i=0;koblas_v4d multiplier={a,a,a,a};
 for(;i<=n-32;i+=32){EIGHT(STEP) EIGHT(STORE)}for(;i<n;++i)v[i]*=a;
}
int main(void){
 void* ob=dlopen("libopenblas.so.0",RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
 void* mk=dlopen("/home/rasmus/.local/share/koblas-onemkl/venv/lib/libmkl_rt.so.3",RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
 if(!ob||!mk){fprintf(stderr,"%s\n",dlerror());return 2;}
 scale_fn f[]={baseline,aligned,unrolled,aligned_unrolled,dlsym(ob,"cblas_dscal"),dlsym(mk,"cblas_dscal")};
 const char*names[]={"baseline","aligned","unrolled","aligned-unrolled","openblas","onemkl"};
 const char*modes[]={"negate","powers-two","reset"};int lengths[]={96,128,192,256};
 puts("n,offset_bytes,mode,implementation,sample,operations,cpu_ns,wall_ns,cpu_ns_per_op,wall_ns_per_op");
 for(int shape=0;shape<4;++shape)for(int offset=0;offset<32;offset+=8)for(int mode=0;mode<3;++mode){
  int n=lengths[shape],batch=262144/n;double* raw; if(posix_memalign((void**)&raw,64,(n+16)*8))abort();
  double*v=(double*)((char*)raw+offset),*initial=malloc(n*8);
  for(int i=0;i<n;++i)initial[i]=(i%251-125)*.00390625;memcpy(v,initial,n*8);
  for(int sample=-2;sample<5;++sample){uint64_t cpu[6]={0},wall[6]={0};
   for(int round=0;round<32;++round)for(int step=0;step<6;++step){int arm=(step+round+sample+6)%6;
    uint64_t w0=now(CLOCK_MONOTONIC),c0=now(CLOCK_THREAD_CPUTIME_ID);
    for(int i=0;i<batch;++i){double a=mode==0?-1.0:(mode==1?(i&1?2.0:.5):.875);if(mode==2)memcpy(v,initial,n*8);f[arm](n,a,v,1);}
    cpu[arm]+=now(CLOCK_THREAD_CPUTIME_ID)-c0;wall[arm]+=now(CLOCK_MONOTONIC)-w0;
   }
   if(sample>=0)for(int arm=0;arm<6;++arm)printf("%d,%d,%s,%s,%d,%d,%lu,%lu,%.17g,%.17g\n",n,offset,modes[mode],names[arm],sample+1,batch*32,cpu[arm],wall[arm],(double)cpu[arm]/(batch*32),(double)wall[arm]/(batch*32));
  }free(raw);free(initial);
 }
}
