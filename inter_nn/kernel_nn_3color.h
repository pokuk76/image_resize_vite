#include <immintrin.h>

inline void kernel
(
 int a,
 int b,
 float* input,
 float* output
 ){

  //kernel operation: 2*2 -> 4*4 (2*4 -> 4*8)
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4;
  __m256 ymm5, ymm6, ymm7, ymm8, ymm9;
  __m256 ymm10, ymm11, ymm12, ymm13, ymm14;
  __m256i ymm15;

  int pixel_len = 12; 
  float* ptr_i = input;
  float* ptr_o = output;

  for (int i = 0; i < a; i += 2){
    for (int j = 0; j < b; j += 2){
      ptr_i = src + i * pixel_len + (2 * 4) * j;
      ptr_o = output + i * (2 * pixel_len) + (2 * (2 * 4)) * j;
  //each pixel is store in BGR, each we view as a float, size of 1 pixel = 3 * 4 = 12 doubles
  //B value
      ymm0 = _mm256_load_ps(ptr_i);
      ymm1 = _mm256_load_ps(ptr_i + 3 * 4);
      ymm0 = _mm256_broadcast_ps(ymm0);
      ymm1 = _mm256_broadcast_ps(ymm1);
      ymm0 = _mm256_permute_ps(ymm0, ymm1, 1 | (2<<4));
      ymm1 = _mm256_permute_ps(ymm0, ymm1, 0 | (2<<4));
      ymm2 = _mm256_load_ps(ptr_i + 2 * 3 * 4);
      ymm3 = _mm256_load_ps(ptr_i + 3 * 3 * 4);
      ymm2 = _mm256_broadcast_ps(ymm2);
      ymm3 = _mm256_broadcast_ps(ymm3);
      ymm2 = _mm256_permute_ps(ymm2, ymm3, 1 | (2<<4));
      ymm3 = _mm256_permute_ps(ymm2, ymm3, 0 | (2<<4));

     //G value 
      ymm4 = _mm256_load_ps(ptr_i + 4);
      ymm5 = _mm256_load_ps(ptr_i + 4 + 3 * 4);
      ymm4 = _mm256_broadcast_ps(ymm4);
      ymm5 = _mm256_broadcast_ps(ymm5);
      ymm4 = _mm256_permute_ps(ymm4, ymm5, 1 | (2<<4));
      ymm5 = _mm256_permute_ps(ymm4, ymm5, 0 | (2<<4));
      ymm6 = _mm256_load_ps(ptr_i + 4 + 2 * 3 * 4);
      ymm7 = _mm256_load_ps(ptr_i + 4 + 3 * 3 * 4);
      ymm6 = _mm256_broadcast_ps(ymm6);
      ymm7 = _mm256_broadcast_ps(ymm7);
      ymm6 = _mm256_permute_ps(ymm6, ymm7, 1 | (2<<4));
      ymm7 = _mm256_permute_ps(ymm6, ymm7, 0 | (2<<4));
      //R value
      ymm8 = _mm256_load_ps(ptr_i + 8);
      ymm9 = _mm256_load_ps(ptr_i + 8);
      ymm8 = _mm256_broadcast_ps(ymm8);
      ymm9 = _mm256_broadcast_ps(ymm9);
      ymm8 = _mm256_permute_ps(ymm8, ymm9, 1 | (2<<4));
      ymm9 = _mm256_permute_ps(ymm8, ymm9, 0 | (2<<4));
      ymm10 = _mm256_load_ps(ptr_i + 8 + 2 * 3 * 4);
      ymm11 = _mm256_load_ps(ptr_i + 8 + 3 * 3 * 4);
      ymm10 = _mm256_broadcast_ps(ymm10);
      ymm11 = _mm256_broadcast_ps(ymm11);
      ymm10 = _mm256_permute_ps(ymm10, ymm11, 1 | (2<<4));
      ymm11 = _mm256_permute_ps(ymm10, ymm11, 0 | (2<<4));

      _mm256_storeu_ps(ptr_o, ymm0);               
      _mm256_storeu_ps(ptr_o + 3 * 4, ymm1);      
      _mm256_storeu_ps(ptr_o + 2 * 3 * 4, ymm2);  
      _mm256_storeu_ps(ptr_o + 3 * 3 * 4, ymm3);  
      _mm256_storeu_ps(ptr_o + 4, ymm4);  
      _mm256_storeu_ps(ptr_o + 4 + 3 * 4, ymm5);  
      _mm256_storeu_ps(ptr_o + 4 + 2 * 3 * 4, ymm6);               
      _mm256_storeu_ps(ptr_o + 4 + 3 * 3 * 4, ymm7);      
      _mm256_storeu_ps(ptr_o + 8, ymm8);  
      _mm256_storeu_ps(ptr_o + 8 + 3 * 4, ymm9);  
      _mm256_storeu_ps(ptr_o + 8 + 2 * 3 * 4, ymm10);  
      _mm256_storeu_ps(ptr_o + 8 + 3 * 3 * 4, ymm11);  
    }
  }
}
