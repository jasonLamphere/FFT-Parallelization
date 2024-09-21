all: serial_fft omp_fft

serial_fft: serial_fft.cpp
	g++ -g -Wall -o serial_fft serial_fft.cpp

omp_fft: omp_fft.cpp
	g++ -g -Wall -fopenmp -o omp_fft omp_fft.cpp -lm

clean: 
	rm omp_fft serial_fft
