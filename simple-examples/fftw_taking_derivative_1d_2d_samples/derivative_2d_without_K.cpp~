// Sample code of Out-of-place 2-D real to complex fast fourier transformation.
// In this code, I take y=sin(x) on [0,2PI], take the transformation
//  and then take its derivative in fourier space, and then came back
//  physicall space!
//  output files: in_data.dat : y=sin(x)cos(2y) on [0,2PI]
//                in_data.dat : y=-2cos(x)sin(2y) on [0,2PI], achieved by FFT

#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>

using namespace std;

inline int I(int i, int j,int Ny){
  return j + Ny*i;
}


int main(){
  
  ofstream out1, out2;
  out1.open("in_data.dat");
  out2.open("out_data.dat");

  //N1 is Number of points, frequency of sampling
  int  Nx=64;
  int  Ny=64;
  int Nyh = Ny/2+1;
  


  // out=> data in complex fourier mode which is complex[Ny*(Nx/2+1)]
  // this size is because of Real To Complex transformation
  fftw_complex *out;
  // in=> real data in physical space, double[Nx*Ny];
  double *in;

  // Lx is the length of domain in x, in this case 2*PI
  // Ly is the length of domain in y, in this case 2*PI
  // We will map this length to 0-2*PI
  double Lx=2.0*M_PI;
  double Ly=2.0*M_PI;

  double *K;
  K = new double[Nx];
  for(int i=0;i<Nx/2;i++) K[i] = 2.0*M_PI/Lx*(double)i;
  for(int i=-Nx/2;i<0;i++) K[i+Nx] = 2.0*M_PI/Lx*(double)i;


  //we have two plans for fft, one going to fourier space
  // and one is for comming back to physical space
  fftw_plan fft_plann,ifft_plann;

  // we are allocaitng memory in this part
  // out is type complex => fftw_complex
  int n1 =Nx*(Ny/2+1);
  out = fftw_alloc_complex(sizeof(fftw_complex)*n1); //out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n1);
  // in is type double
  n1=Nx*Ny;
  in = fftw_alloc_real(sizeof(double)*n1);//(double*) malloc(sizeof(double)*n1);
  


  //initializing the data with function of y=sin(x) for 0<x<2*PI
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      in[I(i,j,Ny)] = sin(Lx*(double)i/(double)Nx) * cos(Ly*(double)j/(double)Ny);
      out1<<Lx*(double)(i)/(double)Nx<<"\t"<<Ly*(double)j/(double)Ny<<"\t"<<in[I(i,j,Ny)]<<"\n";
    }
    out1<<"\n";
  }

  // fft_plan is for going to fourier space, with 2-D real to complex transformation
  fft_plann = fftw_plan_dft_r2c_2d(Nx,Ny, in, out, FFTW_ESTIMATE);


  //running the fft plan
  fftw_execute(fft_plann);

  //here we take the derivative in fourier space,
  // we will multiply each complex number by value of j*i
  // in fourier space we will have f(x_j) = sum{n=-inf,+inf} a_n exp(i*n*x/L*2*PI)
  // in here L = 2PI => f(x_j) = sum{n=-inf,+inf} a_n exp(inx)
  // df/dx = sum (in)* a_n exp(inx)
  double temp;
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      out[I(i,j,Nyh)][0] = -out[I(i,j,Nyh)][0]*(double)j*K[i];//(double)i
      out[I(i,j,Nyh)][1] = -out[I(i,j,Nyh)][1]*(double)j*K[i];//
    }
  }

  // ifft_plan is for comming back to real physical spce with c2r in 2-D
  ifft_plann= fftw_plan_dft_c2r_2d(Nx,Ny,out,in, FFTW_ESTIMATE);  
  // ifft of the function
  fftw_execute(ifft_plann);
  
  // saving the results
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      out2<<Lx*(double)(i)/(double)Nx<<"\t"<<Ly*(double)(j)/(double)Ny<<"\t"<<in[I(i,j,Ny)]/(double)(Nx)/(double)(Ny)<<"\n";
    }
    out2<<"\n";
  }

  //closing the files  
  out1.close();
  out2.close(); 

  
  // removing plans

  fftw_destroy_plan(fft_plann);
  fftw_destroy_plan(ifft_plann);
  fftw_free(in);
  fftw_free(out);


  return 0;

}
