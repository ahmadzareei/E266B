#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>

using namespace std;

inline int I(int i, int j,int Ny){
  return j + Ny*i;
}

void CrankNicolsonDiffusion2D(fftw_complex * in, fftw_complex * out, const int Nx, const int Ny,\
			      const double dt, const double T,const double nux, const double nuy, \
			      double * Kx, double * Ky);

void CrankNicolsonDiffusion2DStep(fftw_complex * in, fftw_complex * out, const int Nx, const int Ny,double * Coeff);

int main(){
  
  ofstream out1, out2;
  out1.open("in_data.dat");
  out2.open("out_data.dat");

  const int Nx = 128;
  const int Ny = 128;

  double * input_data;
  fftw_complex *fft_input_data;
  
  fft_input_data = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  input_data = fftw_alloc_real(sizeof(double)*Nx*Ny);

  const double Lx = 2.0*M_PI;
  const double Ly = 2.0*M_PI;
  const double dt = 0.001;
  const double T =4.0;
  const double nux = 1.0;
  const double nuy =1.0;
  


  double *Kx,*Ky;
  Kx = new double[Nx];
  Ky = new double[Ny];

  for ( int i=0;i<Nx/2;i++) Kx[i] = 2.0*M_PI/Lx*(double)i;
  for ( int i=-Nx/2;i<0;i++) Kx[i+Nx] = 2.0*M_PI/Lx*(double)i;
  Kx[Nx/2]=0.0;

  for ( int i=0;i<Ny/2;i++) Ky[i] = 2.0*M_PI/Ly*(double)i;
  for ( int i=-Ny/2;i<0;i++) Ky[i+Ny] = 2.0*M_PI/Ly*(double)i;
  Ky[Ny/2]=0.0;

  fftw_plan fft_forward_r2c, fft_backward_c2r;
  fft_forward_r2c = fftw_plan_dft_r2c_2d(Nx,Ny,input_data,fft_input_data,FFTW_ESTIMATE);
  fft_backward_c2r = fftw_plan_dft_c2r_2d(Nx,Ny,fft_input_data,input_data,FFTW_ESTIMATE);
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      input_data[I(i,j,Ny)] = sin(Lx*(double)i/(double)Nx) * cos(Ly*(double)j/(double)Ny);
      out1<<Lx*(double)(i)/(double)Nx<<"\t"<<Ly*(double)j/(double)Ny<<"\t"<<input_data[I(i,j,Ny)]<<"\n";
    }
    out1<<"\n";
  }
  fftw_execute(fft_forward_r2c);
  
  CrankNicolsonDiffusion2D(fft_input_data,fft_input_data,Nx,Ny,dt,T,nux,nuy,Kx,Ky);
  
  fftw_execute(fft_backward_c2r);

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      out2<<Lx*(double)(i)/(double)Nx<<"\t"<<Ly*(double)j/(double)Ny<<"\t"<<input_data[I(i,j,Ny)]/(double)Nx/(double)Ny<<"\n";
    }
    out2<<"\n";
  }



  
  
  
  out1.close();
  out2.close();

  fftw_destroy_plan(fft_forward_r2c);
  fftw_destroy_plan(fft_backward_c2r);
  fftw_free(input_data);
  fftw_free(fft_input_data);
  return 0;
}
/*********************************************************************************************************/
//Crank-Nicolson for Diffusion Equation Solver
// in = input, out = output, Nx = x-domain size, Ny = y-domain size,
// dt time step, T= final time, nux = diffusion constant in x direction
// nuy = diffusion constant in y direction
// Kx = frequency coefficient in x direction
// Ky = frequency coefficient in y direction
void CrankNicolsonDiffusion2D(fftw_complex * in, fftw_complex * out, const int Nx, const int Ny,\
			      const double dt, const double T,const double nux, const double nuy, \
			      double * Kx, double * Ky){
  double *Coeff;
  Coeff = new double [Nx*(Ny/2+1)];
  double time;

  // Coeff = Coefficient of (nux n^2 + nuy * m^2) which is repeated in the procedure
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      Coeff[I(i,j,Ny/2+1)] = (1-dt/2.*(Kx[i]*Kx[i]*nux + Ky[j]*Ky[j]*nuy))/(1+ dt/2.*(Kx[i]*Kx[i]*nux + Ky[j]*Ky[j]*nuy));
    }
  }  
  

  time = 0.0;
  while (time<T){
    CrankNicolsonDiffusion2DStep(in,out,Nx,Ny,Coeff);
    time += dt;
  }

}
/****************************************************************************************************/
void CrankNicolsonDiffusion2DStep(fftw_complex * in, fftw_complex * out, const int Nx, const int Ny,double * Coeff){
    for ( int i=0;i<Nx;i++){
      for ( int j=0;j<Ny/2+1;j++){
	out[I(i,j,Ny/2+1)][0] = in[I(i,j,Ny/2+1)][0]*Coeff[I(i,j,Ny/2+1)];
	out[I(i,j,Ny/2+1)][1] = in[I(i,j,Ny/2+1)][1]*Coeff[I(i,j,Ny/2+1)];
      }
    }
}
/*****************************************************************************************************/
