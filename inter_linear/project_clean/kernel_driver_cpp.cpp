#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"

#include "kernel.h"
#define RUNS 5
//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


int main(){
  //kernel size: 6 * 16
  //input image size: a * 3 + 1 rows, b * 8 + 1 cols
  //output image size: 6 * a rows, 16 * b cols
  //output image size should be: 6 * a + 2 rows, 16 * b + 2 cols, but the outermost pixels are corner cases, and I haven't implemented it yet.
  unsigned long long t0, t1, sum1, sum2;
  int a = 100;
  int b = 100;

  //use opencv resize to get a (a*3+1) * (b*8+1) source image
  cv::Mat input_img = cv::imread("512_512Test.PNG");
  cv::Size new_src_size(b * 8 + 1, a * 3 + 1);
  cv::Mat src_img;
  cv::resize(input_img, src_img, new_src_size, 0, 0,cv::INTER_NEAREST);


  //get blue channel, and convert to float array
  cv::Mat blue_channel;
  cv::extractChannel(src_img, blue_channel, 0);
  blue_channel.convertTo(blue_channel, CV_32F);
  float* src = (float*)(blue_channel.data);

  //save src image
  cv::Mat src_img_blue(3*a+1, 8*b+1, CV_32F, src);
  cv::imwrite("src_blue.jpg", src_img_blue);


  float* out;
  posix_memalign((void**) &out, 32, 6*a*16*b * sizeof(float));

  sum1 = 0;
  sum2 = 0;

  for (int runs = 0; runs != RUNS; ++runs){
    t0 = rdtsc();
    kernel(a,b,src,out);
    t1 = rdtsc();
    
    cv::Mat out_img(6*a, 16*b, CV_32F, out);
    cv::imwrite("output_image.jpg", out_img);
    sum1 += (t1 - t0);
  }

  cv::Mat cv_out;
  cv::Size new_src_size1(b * 16 + 2, a * 6 + 2);
  for (int runs = 0; runs != RUNS; ++runs){
    t0 = rdtsc();
    cv::resize(blue_channel, cv_out, new_src_size1, 0, 0,cv::INTER_LINEAR);
    t1 = rdtsc();

    cv::imwrite("output_image_opencv.jpg", cv_out);
    sum2 += (t1 - t0);
  }



  

  printf(" %lf\n", (1.0 * 96 * a * b)/((double)(sum1/(1.0*RUNS))));
  printf(" %lf\n", (1.0 * 96 * a * b)/((double)(sum2/(1.0*RUNS))));
  return 0;
}

