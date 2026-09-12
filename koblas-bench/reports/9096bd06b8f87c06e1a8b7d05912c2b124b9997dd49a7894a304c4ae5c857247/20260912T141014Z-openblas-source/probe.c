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

typedef long BLASLONG;
#define FLOAT double
#include "/home/rasmus/Workspaces/OpenBLAs/kernel/x86_64/dscal_microk_haswell-2.c"
CLONES static void ob_leaf(int n,double a,double*v,int stride){
 (void)stride;if(a==1||n<=0)return;
 int bulk=n&~7;if(bulk)dscal_kernel_8(bulk,&a,v);for(int i=bulk;i<n;++i)v[i]*=a;
}
CLONES static void ob_aligned(int n,double a,double*v,int stride){
 (void)stride;if(a==1||n<=0)return;
 if(n>=128)while((uintptr_t)v&31){*v++*=a;--n;}
 int bulk=n&~7;if(bulk)dscal_kernel_8(bulk,&a,v);for(int i=bulk;i<n;++i)v[i]*=a;
}
#define ALIGN if(n>=128)while((uintptr_t)v&31){*v++*=a;--n;}
#define LOAD(q, dst, at) koblas_v4d dst; __builtin_memcpy(&dst,at+q*4,32);dst*=multiplier;
#define STORE(q, src) __builtin_memcpy(v+q*4,&src,32);
CLONES static void pointers(int n,double a,double*v,int stride){
 (void)stride;if(a==1||n<=0)return;ALIGN
 koblas_v4d multiplier={a,a,a,a};
 for(;n>=16;n-=16,v+=16){
  LOAD(0,a0,v) LOAD(1,a1,v) LOAD(2,a2,v) LOAD(3,a3,v)
  STORE(0,a0) STORE(1,a1) STORE(2,a2) STORE(3,a3)
 }
 for(int i=0;i<n;++i)v[i]*=a;
}
CLONES static void pipeline(int n,double a,double*v,int stride){
 (void)stride;if(a==1||n<=0)return;ALIGN
 koblas_v4d multiplier={a,a,a,a};
 if(n>=16){
  LOAD(0,a0,v) LOAD(1,a1,v) LOAD(2,a2,v) LOAD(3,a3,v)
  for(;n>=32;n-=16,v+=16){
   LOAD(0,b0,v+16) LOAD(1,b1,v+16) LOAD(2,b2,v+16) LOAD(3,b3,v+16)
   STORE(0,a0) STORE(1,a1) STORE(2,a2) STORE(3,a3)
   a0=b0;a1=b1;a2=b2;a3=b3;
  }
  STORE(0,a0) STORE(1,a1) STORE(2,a2) STORE(3,a3)
  n-=16;v+=16;
 }
 for(int i=0;i<n;++i)v[i]*=a;
}
int main(void){
 void* ob=dlopen("libopenblas.so.0",RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
 void* mk=dlopen("/home/rasmus/.local/share/koblas-onemkl/venv/lib/libmkl_rt.so.3",RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
 if(!ob||!mk){fprintf(stderr,"%s\n",dlerror());return 2;}
 scale_fn f[]={baseline,pointers,pipeline,ob_leaf,ob_aligned,dlsym(ob,"cblas_dscal"),dlsym(mk,"cblas_dscal")};
 const char*names[]={"current","pointers","pipeline","openblas-leaf","openblas-aligned","openblas","onemkl"};
 const char*modes[]={"negate","powers-two","reset"};int lengths[]={64,256,4096,65536};
 puts("n,offset_bytes,mode,implementation,sample,operations,cpu_ns,wall_ns,cpu_ns_per_op,wall_ns_per_op");
 for(int shape=0;shape<4;++shape)for(int offset=0;offset<32;offset+=8)for(int mode=0;mode<3;++mode){
  int n=lengths[shape],batch=262144/n;double* raw; if(posix_memalign((void**)&raw,64,(n+16)*8))abort();
  double*v=(double*)((char*)raw+offset),*initial=malloc(n*8);
  for(int i=0;i<n;++i)initial[i]=(i%251-125)*.00390625;memcpy(v,initial,n*8);
  for(int sample=-2;sample<7;++sample){uint64_t cpu[7]={0},wall[7]={0};
   for(int round=0;round<32;++round)for(int step=0;step<7;++step){int arm=(step+round+sample+7)%7;
    uint64_t w0=now(CLOCK_MONOTONIC),c0=now(CLOCK_THREAD_CPUTIME_ID);
    for(int i=0;i<batch;++i){double a=mode==0?-1.0:(mode==1?(i&1?2.0:.5):.875);if(mode==2)memcpy(v,initial,n*8);f[arm](n,a,v,1);}
    cpu[arm]+=now(CLOCK_THREAD_CPUTIME_ID)-c0;wall[arm]+=now(CLOCK_MONOTONIC)-w0;
   }
   if(sample>=0)for(int arm=0;arm<7;++arm)printf("%d,%d,%s,%s,%d,%d,%lu,%lu,%.17g,%.17g\n",n,offset,modes[mode],names[arm],sample+1,batch*32,cpu[arm],wall[arm],(double)cpu[arm]/(batch*32),(double)wall[arm]/(batch*32));
  }free(raw);free(initial);
 }
}
