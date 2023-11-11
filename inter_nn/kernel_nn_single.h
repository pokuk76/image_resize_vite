#include <immintrin.h>


//one color version(?)
inline void kernel
(
 int a,
 int b,
 float* input,
 float* output
 ){

  __m256 ymm0, ymm1, ymm2, ymm3, ymm4;
  __m256 ymm5, ymm6, ymm7, ymm8, ymm9;
  __m256 ymm10, ymm11, ymm12, ymm13, ymm14;
  __m256 ymm15;

  float* ptr_i = input;
  float* ptr_o = output;

  int kernel_r = 6;
  int kernel_c = 4;

  // float a[] = {0,1,2,3,4,5,6,7};
  //   __m256 vec = _mm256_load_ps(a);
  // __m256 shuf1 = _mm256_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 0, 0)); 
  // __m256 shuf2 = _mm256_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 2, 2)); 
  // __m256 result = _mm256_permute2f128_ps(shuf1, shuf2, 0x20); // [0,0,1,1,2,2,3,3]

//Kernel size: 6 * 4 -> 12*8
  for (int i = 0; i < a; i += 6){  //rows
    for (int j = 0; j < b; j += 4){ //cols
      ptr_i = input + i + j;
      ptr_o = output + 2 * i + 2 * j;
    
    //imm reg 1, 2  sect1:3-4-5-0, sec2:7-8-9-6, sec3: 11-12-13-10
      ymm0 = mm256_load_ps(ptr_i);
      ymm1 = _mm256_shuffle_ps(ymm0 ymm0, _MM_SHUFFLE(1, 1, 0, 0));
      ymm2 = _mm256_shuffle_ps(ymm0, ymm0, _MM_SHUFFLE(3, 3, 2, 2));
      ymm3 = _mm256_permute2f128_ps(ymm1, ymm2, 0x20);
      ymm4 = ymm3;
      ymm15 = mm256_load_ps(ptr_i + a);
      ymm1 = _mm256_shuffle_ps(ymm0, ymm0, _MM_SHUFFLE(1, 1, 0, 0));
      ymm2 = _mm256_shuffle_ps(ymm0, ymm0, _MM_SHUFFLE(3, 3, 2, 2));
      ymm5 = _mm256_permute2f128_ps(ymm1, ymm2, 0x20); 
      ymm0 = ymm5;

      ymm6 = mm256_load_ps(ptr_i + 2 * a);
      ymm1 = _mm256_shuffle_ps(ymm6 ymm6, _MM_SHUFFLE(1, 1, 0, 0));
      ymm2 = _mm256_shuffle_ps(ymm6, ymm6, _MM_SHUFFLE(3, 3, 2, 2));
      ymm7 = _mm256_permute2f128_ps(ymm1, ymm2, 0x20); 
      ymm8 = ymm7;
      ymm15 = mm256_load_ps(ptr_i + 3 * a);
      ymm1 = _mm256_shuffle_ps(ymm6, ymm6, _MM_SHUFFLE(1, 1, 0, 0));
      ymm2 = _mm256_shuffle_ps(ymm6, ymm6, _MM_SHUFFLE(3, 3, 2, 2));
      ymm9 = _mm256_permute2f128_ps(ymm1, ymm2, 0x20); 
      ymm6 = ymm9;

      ymm10 = mm256_load_ps(ptr_i + 4 * a);
      ymm1 = _mm256_shuffle_ps(ymm10 ymm10, _MM_SHUFFLE(1, 1, 0, 0));
      ymm2 = _mm256_shuffle_ps(ymm10, ymm10, _MM_SHUFFLE(3, 3, 2, 2));
      ymm11 = _mm256_permute2f128_ps(ymm1, ymm2, 0x20); 
      ymm12 = ymm11;
      ymm15 = mm256_load_ps(ptr_i + 5 * a);
      ymm1 = _mm256_shuffle_ps(ymm10, ymm10, _MM_SHUFFLE(1, 1, 0, 0));
      ymm2 = _mm256_shuffle_ps(ymm10, ymm10, _MM_SHUFFLE(3, 3, 2, 2));
      ymm13 = _mm256_permute2f128_ps(ymm1, ymm2, 0x20); 
      ymm10 = ymm13;

      
      _mm256_storeu_ps(ptr_o, ymm3);  
      _mm256_storeu_ps(ptr_o + a + j, ymm4);      
      _mm256_storeu_ps(ptr_o + 2 * a, ymm5);                   
      _mm256_storeu_ps(ptr_o + 3 * a, ymm0);   

      _mm256_storeu_ps(ptr_o + 4 * a , ymm7);
      _mm256_storeu_ps(ptr_o + 5 * a, ymm8);          
      _mm256_storeu_ps(ptr_o + 6 * a, ymm9);
      _mm256_storeu_ps(ptr_o + 7 * a, ymm6);
      
      _mm256_storeu_ps(ptr_o + 8 * a, ymm11);
      _mm256_storeu_ps(ptr_o + 9 * a, ymm12);  
      _mm256_storeu_ps(ptr_o + 10 * a, ymm13);
      _mm256_storeu_ps(ptr_o + 11 * a, ymm10);  
    }
  }
}
