/* CSCI 551 Final Project On FFT's
 *
 * By Jason K Lamphere
 *
 * Makes use of examples at the following locations
 *
 *
*/


// Iterative serial code retrieved from:
// https://www.nayuki.io/page/free-small-fft-in-multiple-languagesa

//Signal Generation & examples of parallelization
//https://github.com/evanmacbride/parallel-fft

#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::complex;
using std::size_t;
using std::string;
using std::vector;
using std::endl;
using std::cout;


//So much less parameterization with this class based design. Much easier to interpret and read.
class FFT {
  public:

   void twiddle_n_reverse( vector<complex<double>>  &twiddleTable,size_t i ,size_t n, int  & levels, vector<complex<double>>  &vec);
   void fft_plot();
   void signal_generator();
   unsigned long of_two(unsigned long v);
   static size_t reverseBits(size_t x, int n);
   void radix2_fft(std::vector<std::complex<double> > &vec);
   double duration = 1.0;
   int freq = 60;
   unsigned long num_samples;
   int threads;
   vector<complex<double> > signal_buf;
   double step = 0.0;
   string signal_type;
   timespec start_time;
   timespec end_time;

};


 timespec GetElapsed(timespec  ftime , timespec  stime);

/*
 *  Main
 *
 */

int main(int argc, char *argv[]) {
   // Defaults for duration and frequency can be overwritten by command line
   // arguments.
   FFT fft;

   if (argc == 4) {
      fft.signal_type = argv[1];
      fft.duration = atof(argv[2]);
      fft.threads = atof(argv[3]);
   }
   if (argc > 4) {
      fft.signal_type = argv[1];
      fft.duration = atof(argv[2]);
      fft.freq = atoi(argv[3]);
      fft.threads = atof(argv[4]);
   }

   else {
      std::cout << "example: -sine 60 12000 4" << std::endl;
      std::cout << "example: -octaves 60 5000 8"
      std::cout << "example: -noisy 30 1400 16"
      std::cout << "example: signal type, time, frequency, threads"
      exit(0);
   }

   omp_set_num_threads(fft.threads);

   // 44100 represents our sampling frequency.
   //  multiplied by duration is a our total number of samples
   fft.num_samples = (unsigned long)(fft.duration * 44100);
   // however for the radix2 design alg to function it must have a size of the
   // power of 2.
   fft.num_samples = fft.of_two(fft.num_samples);

   // Fill the signal_buffer with a num_samples number of samples spaced a step
   // Writes signal points to file for plotting
   fft.signal_generator();

   timespec sTime, fTime;
   string temps = "";
   if (fft.signal_type == "-sine") {
      temps = "";
   }
   if (fft.signal_type == "-octaves") {
      temps = "octaves ";
   }
   if (fft.signal_type == "-noise") {
      temps = "buried in noise ";
   }
   if (fft.signal_type == "-noisy") {
      temps = "octaves burried in noise ";
   }
   clock_gettime(CLOCK_MONOTONIC, &sTime);
   fft.radix2_fft(fft.signal_buf);
   clock_gettime(CLOCK_MONOTONIC, &fTime);
   cout << "FFT perfomed on " << fft.duration << " second sample of " << fft.freq
             << " cycles per second " << temps << " in " << GetElapsed(sTime,fTime).tv_sec
             << "." << GetElapsed(sTime,fTime).tv_nsec << " seconds, using "
             << fft.threads << " threads" << endl;
   fft.fft_plot();
   return 0;
}  // end main

// FFT
void FFT::radix2_fft(vector<complex<double> > &vec) {
   // Length variables
   size_t n = vec.size();
   int levels = 0;  // Compute levels = floor(log2(n))

   for (size_t temp = n; temp > 1U; temp >>= 1) {
      levels++;
   }
   if (static_cast<size_t>(1U) << levels != n) {
      throw std::domain_error("Length is not a power of 2");
   }
   // Trignometric table
   vector<complex<double> > twiddleTable(n / 2);
   size_t i = 0;

   twiddle_n_reverse( twiddleTable, i, n,levels, vec);



   //butterfly operation variables
   size_t divided_size;
   size_t tablestep;
   complex<double> temp;
   size_t size, j, k;



   // Cooley-Tukey decimation-in-time radix-2 FFT
#pragma omp parallel private(size, divided_size, tablestep, i, j, k, temp) \
    shared(vec)
   {
      for (size = 2; size <= n; size *= 2) {
         divided_size = size / 2;
         tablestep = n / size;
#pragma omp for
         for (i = 0; i < n; i += size) {
            for (j = i, k = 0; j < i + divided_size; j++, k += tablestep) {
               temp = vec[j + divided_size] * twiddleTable[k];
               vec[j + divided_size] = vec[j] - temp;
               vec[j] += temp;
            }
         }
         if (size == n)  // Prevent overflow in 'size *= 2'
            break;
      }
   }
}//end FFT


//bit reversal allows us to do this operation in place. 
 size_t FFT::reverseBits(size_t x, int n) {
   size_t result = 0;
   for (int i = 0; i < n; i++, x >>= 1) result = (result << 1) | (x & 1U);
   return result;
}

//graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
//Rounds up to the next power of two
unsigned long FFT::of_two(unsigned long v) {
   v--;
   v |= v >> 1;
   v |= v >> 2;
   v |= v >> 4;
   v |= v >> 8;
   v |= v >> 16;
   v |= v >> 32;
   v++;
   return v;
}




// In Many of the simpler verions of this algorith this values are generated on the fly in a loop that can't be made parallel easily
// By creating a table of values in advance we take a step that made an unparallel loop take longer and removed it to before where it can be quickly made short work.
//
void FFT::twiddle_n_reverse(  vector<complex<double> >    &twiddleTable,size_t i ,size_t n, int  & levels, vector<complex<double>>  &vec){

#pragma omp parallel for private(i) shared(twiddleTable, n)
   for (i = 0; i < n / 2; i++) {
      twiddleTable[i] = std::polar(1.0, -2 * M_PI * i / n);
   }

// Bit-reversed addressing permutation
// this one is not order dependent so long as the steps are repeated the 
// right number of times. Hence the private i variable 
#pragma omp parallel for private(i) shared(vec, n)
   for (i = 0; i < n; i++) {
      size_t j = reverseBits(i, levels);
      if (j > i) std::swap(vec[i], vec[j]);
   }


}





//I stripped this guy out of a main I found here
//https://github.com/evanmacbride/parallel-fft
//It allows us to generate a signal that can fed into our alg
//while leaving a plot behind for us to graph so we can compare to the final
//FFT graph to make sure it's idenftifying frequencies correctly.
//I didn't want meaningless numbers so the ability to test with real psuedo signals 
//puts alot of context behind the usease of this project.
void FFT::signal_generator() {
   
   FILE *signal_plot = NULL;
   signal_plot = fopen("wave.txt", "w");
   // Default to sine wave
   if (signal_type == " " || signal_type == "-sine") {
      for (unsigned long int  i = 0; i < num_samples; i++) {
         step += duration / num_samples;
         signal_buf.push_back((complex<double>)(sin(2 * M_PI * step * freq)));
         fprintf(signal_plot, "%f\t%g\n", step, real(signal_buf[i]));
      }
      // octaves
   } else if (signal_type == "-octaves") {
      for (unsigned long int i = 0; i < num_samples; i++) {
         step += duration / num_samples;
         signal_buf.push_back(
             (complex<double>)(sin(2 * M_PI * step * freq)) +
             0.33 * (complex<double>)(sin(2 * M_PI * step * 2 * freq)) +
             0.11 * (complex<double>)(sin(2 * M_PI * step * 3 * freq)));
         fprintf(signal_plot, "%f\t%g\n", step, real(signal_buf[i]));
      }
   }
   // Generate noise by using random values
   else if (signal_type == "-noise") {
      int r = 0;
      double randD = 0.0;
      for (unsigned long int i = 0; i < num_samples; i++) {
         step += duration / num_samples;
         r = rand();
         randD = (double)r / INT_MAX;
         signal_buf.push_back((complex<double>)(randD + (r % 3) - 1));
         fprintf(signal_plot, "%f\t%g\n", step, real(signal_buf[i]));
      }
   }
   // Generate a sine with two octaves with noise
   else if (signal_type == "-noisy") {
      int r = 0;
      double randD = 0.0;
      for (unsigned long int i = 0; i < num_samples; i++) {
         step += duration / num_samples;
         r = rand();
         randD = (double)r / INT_MAX;
         signal_buf.push_back(
             (complex<double>)(sin(2 * M_PI * step * freq)) +
             0.33 * (complex<double>)(sin(2 * M_PI * step * 2 * freq)) +
             0.11 * (complex<double>)(sin(2 * M_PI * step * 3 * freq)) +
             0.09 * (complex<double>)(randD + (r % 3) - 1));
         fprintf(signal_plot, "%f\t%g\n", step, real(signal_buf[i]));
      }
   }
   fclose(signal_plot);
}

// Plot the component frequencies represented by the FFT. Adapted from
// Python code found here:
// https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
// This essentially allows us to test our results. Rather than simply perform out FFT forward and backward to make sure the results are the same.
//This lets us visually see how we can make sense of otherwise difficult signal.
void FFT::fft_plot(   ) {
   FILE *fftPlotFile = NULL;
   fftPlotFile = fopen("fft.txt", "w");
   double freq_step = 0.0;
   int end = num_samples / 2;
   double T = duration / num_samples;
   complex<double> val;
   for (int i = 0; i < end; i++) {
      val = (complex<double>)(fabs(signal_buf[i]) / num_samples);
      fprintf(fftPlotFile, "%f\t%g\n", freq_step, real(val));
      freq_step += (1 / T) / num_samples;
   }
   fclose(fftPlotFile);
}

timespec GetElapsed(timespec sTime, timespec fTime) {
   timespec total;
   if ((fTime.tv_nsec - sTime.tv_nsec) < 0) {
      total.tv_sec = fTime.tv_sec - sTime.tv_sec - 1;
      total.tv_nsec = 1000000000 + fTime.tv_nsec - sTime.tv_nsec;

   } else {
      total.tv_sec = fTime.tv_sec - sTime.tv_sec;
      total.tv_nsec = fTime.tv_nsec - sTime.tv_nsec;
   }

   return total;
}
