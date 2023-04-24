#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 fx0vec = _mm256_load_ps(fx);
  __m256 fy0vec = _mm256_load_ps(fy);
  __m256 jvec = _mm256_load_ps(j);
  
  for(int i=0; i<N; i++) {
    //for(int j=0; j<N; j++) {
      //if(i != j) {
        __m256 ivec = _mm256_set1_ps(i);
        __m256 mask = _mm256_cmp_ps(ivec, jvec, _CMP_NEQ_OQ);
        
        __m256 xivec = _mm256_set1_ps(x[i]);
        __m256 yivec = _mm256_set1_ps(y[i]);
        __m256 rxvec = _mm256_sub_ps(xivec, xvec); //float rx = x[i] - x[j];
        __m256 ryvec = _mm256_sub_ps(yivec, yvec); //float ry = y[i] - y[j];

        //float r = std::sqrt(rx * rx + ry * ry);
        __m256 rvec = _mm256_mul_ps(rxvec, rxvec);
        rvec = _mm256_fmadd_ps(ryvec, ryvec, rvec);
        rvec = _mm256_rsqrt_ps(rvec);

        //fx[i] -= rx * m[j] / (r * r * r);
        //fy[i] -= ry * m[j] / (r * r * r);
        __m256 rhsvec = _mm256_mul_ps(mvec, rvec);
        rhsvec = _mm256_mul_ps(rhsvec, rvec);
        rhsvec = _mm256_mul_ps(rhsvec, rvec);
        __m256 fxivec = _mm256_mul_ps(rxvec, rhsvec);
        __m256 fyivec = _mm256_mul_ps(ryvec, rhsvec);
        fxivec = -_mm256_blendv_ps(fx0vec, fxivec, mask);
        fyivec = -_mm256_blendv_ps(fy0vec, fyivec, mask);

        __m256 fxvec = _mm256_permute2f128_ps(fxivec, fxivec, 1);
        fxvec = _mm256_add_ps(fxvec, fxivec);
        fxvec = _mm256_hadd_ps(fxvec, fxvec);
        fxvec = _mm256_hadd_ps(fxvec, fxvec);
        _mm256_store_ps(fx, fxvec);
        
        __m256 fyvec = _mm256_permute2f128_ps(fyivec, fyivec, 1);
        fyvec = _mm256_add_ps(fyvec, fyivec);
        fyvec = _mm256_hadd_ps(fyvec, fyvec);
        fyvec = _mm256_hadd_ps(fyvec, fyvec);
        _mm256_store_ps(fy, fyvec);
      //}
    //}
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
