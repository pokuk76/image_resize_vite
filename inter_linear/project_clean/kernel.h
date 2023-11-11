#include <immintrin.h>




void print_ymm(__m256 ymm) {
  float result[8];
  _mm256_store_ps(result, ymm);
  for (int i = 0; i < 8; i++) {
    printf("%.3f ",result[i]);
  }
}

inline void kernel
(
 int a,
 int b,
 float* src,
 float* output
 ){

  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
  __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14;
  __m256i ymm15 = _mm256_set_epi32(3,3,2,2,1,1,0,0);


  int line_len = 8 * b + 1;
  float* src_ptr = src;
  float* out_ptr = output;


  float TL1[8] = {0.5625,0.1875,0.5625,0.1875,0.5625,0.1875,0.5625,0.1875}; //BL2
  float TL2[8] = {0.1875,0.0625,0.1875,0.0625,0.1875,0.0625,0.1875,0.0625}; //BL1
  float TR1[8] = {0.1875,0.5625,0.1875,0.5625,0.1875,0.5625,0.1875,0.5625}; //BR2
  float TR2[8] = {0.0625,0.1875,0.0625,0.1875,0.0625,0.1875,0.0625,0.1875}; //BR1

  float BL1[8] = {0.1875,0.0625,0.1875,0.0625,0.1875,0.0625,0.1875,0.0625};
  float BL2[8] = {0.5625,0.1875,0.5625,0.1875,0.5625,0.1875,0.5625,0.1875};
  float BR1[8] = {0.0625,0.1875,0.0625,0.1875,0.0625,0.1875,0.0625,0.1875};
  float BR2[8] = {0.1875,0.5625,0.1875,0.5625,0.1875,0.5625,0.1875,0.5625};


  for (int i = 0; i < a; i++){
    for (int j = 0; j < b; j++){
      src_ptr = src + i * 3 * (8 * b + 1) + 8 * j;
      out_ptr = output + i * 6 * 16 * b + 16 * j;
      ymm0 = _mm256_setzero_ps();  ymm1 = _mm256_setzero_ps();
      ymm2 = _mm256_setzero_ps();  ymm3 = _mm256_setzero_ps();
      ymm4 = _mm256_setzero_ps();  ymm5 = _mm256_setzero_ps();
      ymm6 = _mm256_setzero_ps();  ymm7 = _mm256_setzero_ps();
      ymm8 = _mm256_setzero_ps();  ymm9 = _mm256_setzero_ps();
      ymm10 = _mm256_setzero_ps();  ymm11 = _mm256_setzero_ps();


      ymm13 = _mm256_load_ps(TL1);
      ymm14 = _mm256_load_ps(TL2);


      ymm12 = _mm256_loadu_ps(src_ptr);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm0 = _mm256_fmadd_ps(ymm12, ymm13, ymm0);
      ymm1 = _mm256_fmadd_ps(ymm12, ymm14, ymm1);

      

      ymm12 = _mm256_loadu_ps(src_ptr + line_len);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm0 = _mm256_fmadd_ps(ymm12, ymm14, ymm0);
      ymm1 = _mm256_fmadd_ps(ymm12, ymm13, ymm1);
      ymm2 = _mm256_fmadd_ps(ymm12, ymm13, ymm2);
      ymm3 = _mm256_fmadd_ps(ymm12, ymm14, ymm3);
 
      ymm12 = _mm256_loadu_ps(src_ptr + 2 * line_len);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm2 = _mm256_fmadd_ps(ymm12, ymm14, ymm2);
      ymm3 = _mm256_fmadd_ps(ymm12, ymm13, ymm3);
      ymm4 = _mm256_fmadd_ps(ymm12, ymm13, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm12, ymm14, ymm5);

      ymm12 = _mm256_loadu_ps(src_ptr + 3 * line_len);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm4 = _mm256_fmadd_ps(ymm12, ymm14, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm12, ymm13, ymm5);




      ymm12 = _mm256_loadu_ps(src_ptr + 4);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm6 = _mm256_fmadd_ps(ymm12, ymm13, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm12, ymm14, ymm7);

      ymm12 = _mm256_loadu_ps(src_ptr + line_len + 4);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm6 = _mm256_fmadd_ps(ymm12, ymm14, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm12, ymm13, ymm7);
      ymm8 = _mm256_fmadd_ps(ymm12, ymm13, ymm8);
      ymm9 = _mm256_fmadd_ps(ymm12, ymm14, ymm9);

      ymm12 = _mm256_loadu_ps(src_ptr + 2 * line_len + 4);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm8 = _mm256_fmadd_ps(ymm12, ymm14, ymm8);
      ymm9 = _mm256_fmadd_ps(ymm12, ymm13, ymm9);
      ymm10 = _mm256_fmadd_ps(ymm12, ymm13, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm12, ymm14, ymm11);

      ymm12 = _mm256_loadu_ps(src_ptr + 3 * line_len + 4);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm10 = _mm256_fmadd_ps(ymm12, ymm14, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm12, ymm13, ymm11);







      ymm13 = _mm256_load_ps(TR1);
      ymm14 = _mm256_load_ps(TR2);

      ymm12 = _mm256_loadu_ps(src_ptr + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm0 = _mm256_fmadd_ps(ymm12, ymm13, ymm0);
      ymm1 = _mm256_fmadd_ps(ymm12, ymm14, ymm1);

      ymm12 = _mm256_loadu_ps(src_ptr + line_len + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm0 = _mm256_fmadd_ps(ymm12, ymm14, ymm0);
      ymm1 = _mm256_fmadd_ps(ymm12, ymm13, ymm1);
      ymm2 = _mm256_fmadd_ps(ymm12, ymm13, ymm2);
      ymm3 = _mm256_fmadd_ps(ymm12, ymm14, ymm3);

      ymm12 = _mm256_loadu_ps(src_ptr + 2 * line_len + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm2 = _mm256_fmadd_ps(ymm12, ymm14, ymm2);
      ymm3 = _mm256_fmadd_ps(ymm12, ymm13, ymm3);
      ymm4 = _mm256_fmadd_ps(ymm12, ymm13, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm12, ymm14, ymm5);

      ymm12 = _mm256_loadu_ps(src_ptr + 3 * line_len + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm4 = _mm256_fmadd_ps(ymm12, ymm14, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm12, ymm13, ymm5);


      ymm12 = _mm256_loadu_ps(src_ptr + 4 + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm6 = _mm256_fmadd_ps(ymm12, ymm13, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm12, ymm14, ymm7);

      ymm12 = _mm256_loadu_ps(src_ptr + line_len + 4 + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm6 = _mm256_fmadd_ps(ymm12, ymm14, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm12, ymm13, ymm7);
      ymm8 = _mm256_fmadd_ps(ymm12, ymm13, ymm8);
      ymm9 = _mm256_fmadd_ps(ymm12, ymm14, ymm9);

      ymm12 = _mm256_loadu_ps(src_ptr + 2 * line_len + 4 + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm8 = _mm256_fmadd_ps(ymm12, ymm14, ymm8);
      ymm9 = _mm256_fmadd_ps(ymm12, ymm13, ymm9);
      ymm10 = _mm256_fmadd_ps(ymm12, ymm13, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm12, ymm14, ymm11);

      ymm12 = _mm256_loadu_ps(src_ptr + 3 * line_len + 4 + 1);
      ymm12 = _mm256_permutevar8x32_ps(ymm12, ymm15); //00112233
      ymm10 = _mm256_fmadd_ps(ymm12, ymm14, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm12, ymm13, ymm11);

      //printf("%d %d\n",i,j);
    /*
      printf("\n");
      printf("\n");
      printf("\n");
      print_ymm(ymm0); print_ymm(ymm6); printf("\n");
      print_ymm(ymm1); print_ymm(ymm7); printf("\n");
      print_ymm(ymm2); print_ymm(ymm8); printf("\n");
      print_ymm(ymm3); print_ymm(ymm9); printf("\n");
      print_ymm(ymm4); print_ymm(ymm10); printf("\n");
      print_ymm(ymm5); print_ymm(ymm11); printf("\n");
    */
    _mm256_storeu_ps(out_ptr, ymm0);               _mm256_storeu_ps(out_ptr + 8, ymm6);
    _mm256_storeu_ps(out_ptr + b * 16, ymm1);      _mm256_storeu_ps(out_ptr + b * 16 + 8, ymm7);
    _mm256_storeu_ps(out_ptr + 2 * b * 16, ymm2);  _mm256_storeu_ps(out_ptr + 2 * b * 16 + 8, ymm8);
    _mm256_storeu_ps(out_ptr + 3 * b * 16, ymm3);  _mm256_storeu_ps(out_ptr + 3 * b * 16 + 8, ymm9);
    _mm256_storeu_ps(out_ptr + 4 * b * 16, ymm4);  _mm256_storeu_ps(out_ptr + 4 * b * 16 + 8, ymm10);
    _mm256_storeu_ps(out_ptr + 5 * b * 16, ymm5);  _mm256_storeu_ps(out_ptr + 5 * b * 16 + 8, ymm11);

    }
  }
}
