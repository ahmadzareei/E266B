#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>
using namespace std;
//********************************Index counter
inline int I(int i, int j,int Ny){
  return j + Ny*i;
}
//******************************************************************
void fft_forward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V);
void fft_backward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V);
void zero_out( fftw_complex * V,int Nx, int Ny);
void energy_computation(int Nx, int Ny, double* Vx, double*Vy);
//********************************** Initial Data ************************************************
const int Nx = 128;
const int Ny = 128;
const double Lx = 2.0*M_PI;
const double Ly = 2.0*M_PI;
const double dt = 0.01;
const double T = 0.5;
double *Kx,*Ky;
double nu = .0;
//**************************************************************************************************


int main(){
  cout.precision(15);
  cout<<"Opening File to write the data....\n";
  ofstream test;
  test.open("test.dat");
  ofstream out2;
  out2.open("energy.dat");
  

  cout<<"Initializing the data .....\n";
  //************************Initializing Initial Data*******************************************************************************************
  Kx = new double[Nx];
  Ky = new double[Ny];
  
  for ( int i=0;i<Nx/2;i++) Kx[i] = 2.0*M_PI*(double)i/Lx;
  for ( int i=-Nx/2;i<0;i++) Kx[i+Nx] = 2.0*M_PI*(double)i/Lx;
  Kx[Nx/2]=0.0;
  
  for ( int i=0;i<Ny/2;i++) Ky[i] = 2.0*M_PI*(double)i/Ly;
  for ( int i=-Ny/2;i<0;i++) Ky[i+Ny] = 2.0*M_PI*(double)i/Ly;
  Ky[Ny/2]=0.0;
  
  cout<<"Allocating the memory ....\n";
  //************************Allocating Memory****************************************************************************************************************
  fftw_complex *Vx_N,  *Vy_N ,*Vx_N1, *Vy_N1 , *Wz_N , *Wz_N1;
  fftw_complex *PI_N, *PI_N1, *VWx_N, *VWx_N1,*VWy_N, *VWy_N1;
  
  fftw_complex * Vx_temp, *Vy_temp;
  
  double *d_Vx_N,  *d_Vy_N ,*d_Vx_N1, *d_Vy_N1,*d_Wz_N,*d_Wz_N1;
  double  *d_PI_N, *d_PI_N1 , *d_VWx_N, *d_VWx_N1,*d_VWy_N, *d_VWy_N1 ;

  d_Vx_N = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_Vy_N = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_Vx_N1 = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_Vy_N1 = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_PI_N = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_PI_N1 = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_Wz_N = fftw_alloc_real(sizeof(double)*Nx*Ny);  
  d_Wz_N1 = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_VWx_N = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_VWx_N1 = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_VWy_N = fftw_alloc_real(sizeof(double)*Nx*Ny);
  d_VWy_N1 = fftw_alloc_real(sizeof(double)*Nx*Ny);


  Vx_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  Vy_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  Vx_N1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  Vy_N1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  PI_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  PI_N1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  Wz_N1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  Wz_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  VWx_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));  
  VWx_N1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  VWy_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));  
  VWy_N1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  Vx_temp = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  Vy_temp = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));
  //***********************Making FFT Transforms***************************************************************************************************************
  fftw_plan forward_Vx_N, backward_Vx_N,forward_Vy_N, backward_Vy_N, forward_Wz_N, backward_Wz_N, forward_PI_N, backward_PI_N, forward_VW_N, backward_VW_N;
  fftw_plan forward_Vx_N1, backward_Vx_N1,forward_Vy_N1, backward_Vy_N1, forward_Wz_N1, backward_Wz_N1, forward_PI_N1, backward_PI_N1, forward_VW_N1, backward_VW_N1;
  
  fftw_plan forward_VWx_N,backward_VWx_N,forward_VWy_N,backward_VWy_N,forward_VWx_N1,backward_VWx_N1,forward_VWy_N1,backward_VWy_N1;
  
  forward_Vx_N = fftw_plan_dft_r2c_2d(Nx,Ny,d_Vx_N,Vx_N,FFTW_ESTIMATE);
  backward_Vx_N = fftw_plan_dft_c2r_2d(Nx,Ny,Vx_N,d_Vx_N,FFTW_ESTIMATE);

  forward_Vy_N = fftw_plan_dft_r2c_2d(Nx,Ny,d_Vy_N,Vy_N,FFTW_ESTIMATE);
  backward_Vy_N = fftw_plan_dft_c2r_2d(Nx,Ny,Vy_N,d_Vy_N,FFTW_ESTIMATE);

  forward_Wz_N = fftw_plan_dft_r2c_2d(Nx,Ny,d_Wz_N,Wz_N,FFTW_ESTIMATE);
  backward_Wz_N = fftw_plan_dft_c2r_2d(Nx,Ny,Wz_N,d_Wz_N,FFTW_ESTIMATE);
  
  forward_PI_N = fftw_plan_dft_r2c_2d(Nx,Ny,d_PI_N,PI_N,FFTW_ESTIMATE);
  backward_PI_N = fftw_plan_dft_c2r_2d(Nx,Ny,PI_N,d_PI_N,FFTW_ESTIMATE);

  forward_VWx_N = fftw_plan_dft_r2c_2d(Nx,Ny,d_VWx_N,VWx_N,FFTW_ESTIMATE);
  backward_VWx_N = fftw_plan_dft_c2r_2d(Nx,Ny,VWx_N,d_VWx_N,FFTW_ESTIMATE);

  forward_VWy_N = fftw_plan_dft_r2c_2d(Nx,Ny,d_VWy_N,VWy_N,FFTW_ESTIMATE);
  backward_VWy_N = fftw_plan_dft_c2r_2d(Nx,Ny,VWy_N,d_VWy_N,FFTW_ESTIMATE);


  forward_Vx_N1 = fftw_plan_dft_r2c_2d(Nx,Ny,d_Vx_N1,Vx_N1,FFTW_ESTIMATE);
  backward_Vx_N1 = fftw_plan_dft_c2r_2d(Nx,Ny,Vx_N1,d_Vx_N1,FFTW_ESTIMATE);

  forward_Vy_N1 = fftw_plan_dft_r2c_2d(Nx,Ny,d_Vy_N1,Vy_N1,FFTW_ESTIMATE);
  backward_Vy_N1 = fftw_plan_dft_c2r_2d(Nx,Ny,Vy_N1,d_Vy_N1,FFTW_ESTIMATE);

  forward_Wz_N1 = fftw_plan_dft_r2c_2d(Nx,Ny,d_Wz_N1,Wz_N1,FFTW_ESTIMATE);
  backward_Wz_N1 = fftw_plan_dft_c2r_2d(Nx,Ny,Wz_N1,d_Wz_N1,FFTW_ESTIMATE);
  
  forward_PI_N1 = fftw_plan_dft_r2c_2d(Nx,Ny,d_PI_N1,PI_N1,FFTW_ESTIMATE);
  backward_PI_N1 = fftw_plan_dft_c2r_2d(Nx,Ny,PI_N1,d_PI_N1,FFTW_ESTIMATE);

  forward_VWx_N1 = fftw_plan_dft_r2c_2d(Nx,Ny,d_VWx_N1,VWx_N1,FFTW_ESTIMATE);
  backward_VWx_N1 = fftw_plan_dft_c2r_2d(Nx,Ny,VWx_N1,d_VWx_N1,FFTW_ESTIMATE); 

  forward_VWy_N1 = fftw_plan_dft_r2c_2d(Nx,Ny,d_VWy_N1,VWy_N1,FFTW_ESTIMATE);
  backward_VWy_N1 = fftw_plan_dft_c2r_2d(Nx,Ny,VWy_N1,d_VWy_N1,FFTW_ESTIMATE);
  //********************************************************************************************************************************
  
  cout<<"Initializing the initial condition...\n";
  // ******************************** Initial Condition ****************************************************************************  
  //Setting Pressure and Initial Velocity
  for(int i = 0; i< Nx;i++){
    for(int j=0;j<Ny;j++){
      d_Vx_N1[I(i,j,Ny)] = sin(Ly*(double)j/(double)Ny);
      d_Vy_N1[I(i,j,Ny)] = 0.0;
      d_PI_N1[I(i,j,Ny)] = 0.0 + 0.5* (d_Vx_N1[I(i,j,Ny)]*d_Vx_N1[I(i,j,Ny)] + d_Vy_N1[I(i,j,Ny)]*d_Vy_N1[I(i,j,Ny)]) ;
    }
  }
    
  cout<<"Doing the first step using Runge Kutta 2second Order...\n";
  //**********************************************************************************************************************************
  //***************************Doing first step ******* Using Richardson Extrapolation ***********************************************
  //*********************************This part computes f(t1,y1) and then computes v1+0.5*h*f(t1,y1) *********************************

  fft_forward(forward_Vx_N1,Nx,Ny,d_Vx_N1,Vx_N1);
  fft_forward(forward_Vy_N1,Nx,Ny,d_Vy_N1,Vy_N1);
  fft_forward(forward_PI_N1,Nx,Ny,d_PI_N1, PI_N1);

  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      Wz_N1[I(i,j,Ny/2+1)][0] = -Kx[i]*Vy_N1[I(i,j,Ny/2+1)][1] + Ky[j]*Vx_N1[I(i,j,Ny/2+1)][1];
      Wz_N1[I(i,j,Ny/2+1)][1] =  Kx[i]*Vy_N1[I(i,j,Ny/2+1)][0] - Ky[j]*Vx_N1[I(i,j,Ny/2+1)][0];
    }
  }

  fft_backward(backward_Wz_N1,Nx,Ny,d_Wz_N1, Wz_N1);

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      d_VWx_N1[I(i,j,Ny)] =   d_Vy_N1[I(i,j,Ny)] * d_Wz_N1[I(i,j,Ny)];
      d_VWy_N1[I(i,j,Ny)] = - d_Vx_N1[I(i,j,Ny)] * d_Wz_N1[I(i,j,Ny)]; 
    }
  }
  
  fft_forward(forward_VWx_N1,Nx,Ny,d_VWx_N1,VWx_N1);
  fft_forward(forward_VWy_N1,Nx,Ny,d_VWy_N1,VWy_N1);

  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      Vx_temp[I(i,j,Ny/2+1)][0] = VWx_N1[I(i,j,Ny/2+1)][0]  + Kx[i] * PI_N1[I(i,j,Ny/2+1)][1] - nu * Kx[i]*Kx[i] * Vx_N1[I(i,j,Ny/2+1)][0];
      Vx_temp[I(i,j,Ny/2+1)][1] = VWx_N1[I(i,j,Ny/2+1)][1]  - Kx[i] * PI_N1[I(i,j,Ny/2+1)][0] - nu * Kx[i]*Kx[i] * Vx_N1[I(i,j,Ny/2+1)][0];

      Vy_temp[I(i,j,Ny/2+1)][0] = VWy_N1[I(i,j,Ny/2+1)][0]  + Ky[j] * PI_N1[I(i,j,Ny/2+1)][1] - nu * Ky[j]*Ky[j] * Vy_N1[I(i,j,Ny/2+1)][0];
      Vy_temp[I(i,j,Ny/2+1)][1] = VWy_N1[I(i,j,Ny/2+1)][1]  - Ky[j] * PI_N1[I(i,j,Ny/2+1)][0] - nu * Ky[j]*Ky[j] * Vy_N1[I(i,j,Ny/2+1)][0];
    }
  }


  
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      Vx_N[I(i,j,Ny/2+1)][0] = Vx_N1[I(i,j,Ny/2+1)][0] + 0.5*dt*Vx_temp[I(i,j,Ny/2+1)][0];
      Vx_N[I(i,j,Ny/2+1)][1] = Vx_N1[I(i,j,Ny/2+1)][1] + 0.5*dt*Vx_temp[I(i,j,Ny/2+1)][1];

      Vy_N[I(i,j,Ny/2+1)][0] = Vy_N1[I(i,j,Ny/2+1)][0] + 0.5*dt*Vy_temp[I(i,j,Ny/2+1)][0];
      Vy_N[I(i,j,Ny/2+1)][1] = Vy_N1[I(i,j,Ny/2+1)][1] + 0.5*dt*Vy_temp[I(i,j,Ny/2+1)][1];
    }
  }
  
  fft_backward(backward_Vx_N,Nx,Ny,d_Vx_N,Vx_N);
  fft_backward(backward_Vy_N,Nx,Ny,d_Vy_N,Vy_N);  


  
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny;j++){
      d_PI_N[I(i,j,Ny)] = d_PI_N1[I(i,j,Ny)] - 0.5*d_Vx_N1[I(i,j,Ny)]*d_Vx_N1[I(i,j,Ny)] - 0.5*d_Vy_N1[I(i,j,Ny)]*d_Vy_N1[I(i,j,Ny)] + 0.5*d_Vx_N[I(i,j,Ny)]*d_Vx_N[I(i,j,Ny)] + 0.5*d_Vy_N[I(i,j,Ny)]*d_Vy_N[I(i,j,Ny)]; 
    }
  }
  
  //********************This part computes V_2 = V_1 + h f(t+1/2h, V1 + 0.5*h*f(t1,y1)) *******
  // and put value of V2 in V_N 
  
  fft_forward(forward_Vx_N,Nx,Ny,d_Vx_N,Vx_N);
  fft_forward(forward_Vy_N,Nx,Ny,d_Vy_N,Vy_N);
  fft_forward(forward_PI_N1,Nx,Ny,d_PI_N, PI_N);

  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      Wz_N[I(i,j,Ny/2+1)][0] = -Kx[i]*Vy_N[I(i,j,Ny/2+1)][1] + Ky[j]*Vx_N[I(i,j,Ny/2+1)][1];
      Wz_N[I(i,j,Ny/2+1)][1] =  Kx[i]*Vy_N[I(i,j,Ny/2+1)][0] - Ky[j]*Vx_N[I(i,j,Ny/2+1)][0];
    }
  }

  fft_backward(backward_Wz_N,Nx,Ny,d_Wz_N, Wz_N);

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      d_VWx_N[I(i,j,Ny)] =   d_Vy_N[I(i,j,Ny)] * d_Wz_N[I(i,j,Ny)];
      d_VWy_N[I(i,j,Ny)] = - d_Vx_N[I(i,j,Ny)] * d_Wz_N[I(i,j,Ny)]; 
    }
  }
  
  fft_forward(forward_VWx_N,Nx,Ny,d_VWx_N,VWx_N);
  fft_forward(forward_VWy_N,Nx,Ny,d_VWy_N,VWy_N);

  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      Vx_temp[I(i,j,Ny/2+1)][0] = VWx_N[I(i,j,Ny/2+1)][0]  + Kx[i] * PI_N[I(i,j,Ny/2+1)][1] - nu * Kx[i]*Kx[i] * Vx_N[I(i,j,Ny/2+1)][0];
      Vx_temp[I(i,j,Ny/2+1)][1] = VWx_N[I(i,j,Ny/2+1)][1]  - Kx[i] * PI_N[I(i,j,Ny/2+1)][0] - nu* Kx[i]*Kx[i] * Vx_N[I(i,j,Ny/2+1)][0];

      Vy_temp[I(i,j,Ny/2+1)][0] = VWy_N[I(i,j,Ny/2+1)][0]  + Ky[j] * PI_N[I(i,j,Ny/2+1)][1] - nu * Ky[j]*Ky[j] * Vy_N[I(i,j,Ny/2+1)][0];
      Vy_temp[I(i,j,Ny/2+1)][1] = VWy_N[I(i,j,Ny/2+1)][1]  - Ky[j] * PI_N[I(i,j,Ny/2+1)][0] - nu * Ky[j]*Ky[j] * Vy_N[I(i,j,Ny/2+1)][0];
    }
  }

  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      Vx_N[I(i,j,Ny/2+1)][0] = Vx_N1[I(i,j,Ny/2+1)][0] + dt*Vx_temp[I(i,j,Ny/2+1)][0];
      Vx_N[I(i,j,Ny/2+1)][1] = Vx_N1[I(i,j,Ny/2+1)][1] + dt*Vx_temp[I(i,j,Ny/2+1)][1];

      Vy_N[I(i,j,Ny/2+1)][0] = Vy_N1[I(i,j,Ny/2+1)][0] + dt*Vy_temp[I(i,j,Ny/2+1)][0];
      Vy_N[I(i,j,Ny/2+1)][1] = Vy_N1[I(i,j,Ny/2+1)][1] + dt*Vy_temp[I(i,j,Ny/2+1)][1];
    }
  }

  cout<<"Starting the iteration for the reminded time... \n";
  //***********************************************************************
  //*************So far we have values of V_N-1 and V_N *******************
  //************* at this part we will do the fractional step thing *******
  double energy = 0.0;
  for(int iteration = 0; iteration <1000;iteration++){
    
    //computing W_N
    for ( int i=0;i<Nx;i++){
      for ( int j=0;j<Ny/2+1;j++){
	Wz_N[I(i,j,Ny/2+1)][0] = -Kx[i]*Vy_N[I(i,j,Ny/2+1)][1] + Ky[j]*Vx_N[I(i,j,Ny/2+1)][1];
	Wz_N[I(i,j,Ny/2+1)][1] =  Kx[i]*Vy_N[I(i,j,Ny/2+1)][0] - Ky[j]*Vx_N[I(i,j,Ny/2+1)][0];
      }
    }
    
    fft_backward(backward_Wz_N,Nx,Ny,d_Wz_N,Wz_N);
    fft_backward(backward_Vx_N,Nx,Ny,d_Vx_N,Vx_N);
    fft_backward(backward_Vy_N,Nx,Ny,d_Vy_N,Vy_N);
    
    //Computing the value of energy
    //energy_computation(Nx,Ny,d_Vx_N,d_Vy_N);
    
    energy = 0.0;
    for (int i=0;i<Nx;i++){
      for(int j=0;j<Ny;j++){
	energy = 0.5 * (d_Vx_N[I(i,j,Ny)]*d_Vx_N[I(i,j,Ny)] + d_Vy_N[I(i,j,Ny)]*d_Vy_N[I(i,j,Ny)]);
      }
    }
    
    out2<<iteration*dt<<"\t"<<fixed<<energy<<"\n";



    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny;j++){
	d_VWx_N[I(i,j,Ny)] =   d_Vy_N[I(i,j,Ny)] * d_Wz_N[I(i,j,Ny)];
	d_VWy_N[I(i,j,Ny)] = - d_Vx_N[I(i,j,Ny)] * d_Wz_N[I(i,j,Ny)]; 
      }
    }
    
    fft_forward(forward_VWx_N,Nx,Ny,d_VWx_N,VWx_N);
    fft_forward(forward_VWy_N,Nx,Ny,d_VWy_N,VWy_N);
    fft_forward(forward_Vx_N,Nx,Ny,d_Vx_N,Vx_N);
    fft_forward(forward_Vy_N,Nx,Ny,d_Vy_N,Vy_N);
    
    //Computing Vbar^{n+1/2}
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny/2+1;j++){
	Vx_temp[I(i,j,Ny/2+1)][0] = Vx_N[I(i,j,Ny/2+1)][0] + dt/2.0*(3.0*VWx_N[I(i,j,Ny/2+1)][0] - VWx_N1[I(i,j,Ny/2+1)][0]) + dt/2.0* (-Kx[i])*PI_N1[I(i,j,Ny/2+1)][1];
	Vx_temp[I(i,j,Ny/2+1)][1] = Vx_N[I(i,j,Ny/2+1)][1] + dt/2.0*(3.0*VWx_N[I(i,j,Ny/2+1)][1] - VWx_N1[I(i,j,Ny/2+1)][1]) + dt/2.0* ( Kx[i])*PI_N1[I(i,j,Ny/2+1)][0];
	
	Vy_temp[I(i,j,Ny/2+1)][0] = Vy_N[I(i,j,Ny/2+1)][0] + dt/2.0*(3.0*VWy_N[I(i,j,Ny/2+1)][0] - VWy_N1[I(i,j,Ny/2+1)][0]) + dt/2.0* (-Ky[j])*PI_N1[I(i,j,Ny/2+1)][1];
	Vy_temp[I(i,j,Ny/2+1)][1] = Vy_N[I(i,j,Ny/2+1)][1] + dt/2.0*(3.0*VWy_N[I(i,j,Ny/2+1)][1] - VWy_N1[I(i,j,Ny/2+1)][1]) + dt/2.0* ( Ky[j])*PI_N1[I(i,j,Ny/2+1)][0];
      }
    }
    
    
    // Computing PI_N, presure at N such that 3/2 dt D^2 PI_N = Div Vbar^{N+1/2}
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny/2+1;j++){
	if( (i!=0)||(j!=0)){
	  if((i!=Nx/2) && (j!=Ny/2)){ 
	    PI_N[I(i,j,Ny/2+1)][0] = 2.0/(-3.0*dt*(Kx[i]*Kx[i] + Ky[j]*Ky[j]))*(-Kx[i]*Vx_temp[I(i,j,Ny/2+1)][1] - Ky[j]*Vy_temp[I(i,j,Ny/2+1)][1]) ;
	    PI_N[I(i,j,Ny/2+1)][1] = 2.0/(-3.0*dt*(Kx[i]*Kx[i] + Ky[j]*Ky[j]))*( Kx[i]*Vx_temp[I(i,j,Ny/2+1)][0] + Ky[j]*Vy_temp[I(i,j,Ny/2+1)][0]) ;
	  }
	}
      }
    }
    
    PI_N[I(0,0,Ny/2+1)][0] = 0.0;
    PI_N[I(0,0,Ny/2+1)][1] = 0.0;
    zero_out(PI_N,Nx,Ny);
    
    
    
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny/2+1;j++){
	Vx_temp[I(i,j,Ny/2+1)][0] = Vx_temp[I(i,j,Ny/2+1)][0] - 3.0/2.0*dt*(-Kx[i])*PI_N[I(i,j,Ny/2+1)][1];
	Vx_temp[I(i,j,Ny/2+1)][1] = Vx_temp[I(i,j,Ny/2+1)][1] - 3.0/2.0*dt*( Kx[i])*PI_N[I(i,j,Ny/2+1)][0];
	
	Vy_temp[I(i,j,Ny/2+1)][0] = Vy_temp[I(i,j,Ny/2+1)][0] - 3.0/2.0*dt*(-Ky[j])*PI_N[I(i,j,Ny/2+1)][1];
	Vy_temp[I(i,j,Ny/2+1)][1] = Vy_temp[I(i,j,Ny/2+1)][1] - 3.0/2.0*dt*( Ky[j])*PI_N[I(i,j,Ny/2+1)][0];
      }
    }
    
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny/2+1;j++){
	Vx_temp[I(i,j,Ny/2+1)][0] = (Vx_temp[I(i,j,Ny/2+1)][0] - nu*dt/2.0*(Kx[i]*Kx[i]+Ky[j]*Ky[j])*Vx_N[I(i,j,Ny/2+1)][0])/(1.0 +(Kx[i]*Kx[i]+Ky[j]*Ky[j])*nu*dt/2.0);
	Vx_temp[I(i,j,Ny/2+1)][1] = (Vx_temp[I(i,j,Ny/2+1)][1] - nu*dt/2.0*(Kx[i]*Kx[i]+Ky[j]*Ky[j])*Vx_N[I(i,j,Ny/2+1)][1])/(1.0 +(Kx[i]*Kx[i]+Ky[j]*Ky[j])*nu*dt/2.0);
	
	Vy_temp[I(i,j,Ny/2+1)][0] = (Vy_temp[I(i,j,Ny/2+1)][0] - nu*dt/2.0*(Kx[i]*Kx[i]+Ky[j]*Ky[j])*Vy_N[I(i,j,Ny/2+1)][0])/(1.0 +(Kx[i]*Kx[i]+Ky[j]*Ky[j])*nu*dt/2.0);
	Vy_temp[I(i,j,Ny/2+1)][1] = (Vy_temp[I(i,j,Ny/2+1)][1] - nu*dt/2.0*(Kx[i]*Kx[i]+Ky[j]*Ky[j])*Vy_N[I(i,j,Ny/2+1)][1])/(1.0 +(Kx[i]*Kx[i]+Ky[j]*Ky[j])*nu*dt/2.0);
      }
    }
    
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny/2+1;j++){
	Vx_N1[I(i,j,Ny/2+1)][0] = Vx_N[I(i,j,Ny/2+1)][0];
	Vx_N1[I(i,j,Ny/2+1)][1] = Vx_N[I(i,j,Ny/2+1)][1];
	Vy_N1[I(i,j,Ny/2+1)][0] = Vy_N[I(i,j,Ny/2+1)][0];
	Vy_N1[I(i,j,Ny/2+1)][1] = Vy_N[I(i,j,Ny/2+1)][1];
	
	Vx_N[I(i,j,Ny/2+1)][0] = Vx_temp[I(i,j,Ny/2+1)][0];
	Vx_N[I(i,j,Ny/2+1)][1] = Vx_temp[I(i,j,Ny/2+1)][1];
	Vy_N[I(i,j,Ny/2+1)][0] = Vy_temp[I(i,j,Ny/2+1)][0];
	Vy_N[I(i,j,Ny/2+1)][1] = Vy_temp[I(i,j,Ny/2+1)][1];
	
	PI_N1[I(i,j,Ny/2+1)][0] = PI_N[I(i,j,Ny/2+1)][0];
	PI_N1[I(i,j,Ny/2+1)][1] = PI_N[I(i,j,Ny/2+1)][1];
	
	VWx_N1[I(i,j,Ny/2+1)][0] = VWx_N[I(i,j,Ny/2+1)][0];
	VWx_N1[I(i,j,Ny/2+1)][1] = VWx_N[I(i,j,Ny/2+1)][1];
	VWy_N1[I(i,j,Ny/2+1)][0] = VWy_N[I(i,j,Ny/2+1)][0];
	VWy_N1[I(i,j,Ny/2+1)][1] = VWy_N[I(i,j,Ny/2+1)][1];
      }
    }

    
  }

  //***************************************FOR TESTING ****************************** 
  //Now lets first take a look at values of second step

  fft_backward(backward_Vx_N,Nx,Ny,d_Vx_N,Vx_N);
  fft_backward(backward_Vy_N,Nx,Ny,d_Vy_N,Vy_N);
  //fft_backward(backward_PI_N,Nx,Ny,d_PI_N,PI_N);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      test<<Lx*(double)(i)/(double)Nx<<"\t"<<Ly*(double)j/(double)Ny<<"\t"<<d_Vx_N[I(i,j,Ny)]<<"\n";//d_PI_N[I(i,j,Ny)]<<"\n";//
    }
    test<<"\n";
  }
  //****************************************DEBUG MODE *********************************




  fftw_destroy_plan(forward_Vx_N);
  fftw_destroy_plan(backward_Vx_N);
  fftw_destroy_plan(forward_Vy_N);
  fftw_destroy_plan(backward_Vy_N);
  fftw_destroy_plan(forward_Vx_N1);
  fftw_destroy_plan(backward_Vx_N1);
  fftw_destroy_plan(forward_Vy_N1);
  fftw_destroy_plan(backward_Vy_N1);
  fftw_destroy_plan(forward_VWx_N);
  fftw_destroy_plan(backward_VWx_N);
  fftw_destroy_plan(forward_VWy_N);
  fftw_destroy_plan(backward_VWy_N);
  fftw_destroy_plan(forward_Wz_N1);
  fftw_destroy_plan(backward_Wz_N1);
  fftw_destroy_plan(forward_Wz_N);
  fftw_destroy_plan(backward_Wz_N);
  fftw_destroy_plan(forward_PI_N);
  fftw_destroy_plan(backward_PI_N);
  fftw_destroy_plan(forward_PI_N1);
  fftw_destroy_plan(backward_PI_N1);
  fftw_destroy_plan(forward_VWx_N1);
  fftw_destroy_plan(backward_VWx_N1);
  fftw_destroy_plan(forward_VWy_N1);
  fftw_destroy_plan(backward_VWy_N1);
  
  fftw_free(d_Vx_N);
  fftw_free(d_Vy_N);
  fftw_free( d_Vx_N1);
  fftw_free(d_Vy_N1);
  fftw_free(d_PI_N);
  fftw_free(d_PI_N1);
  fftw_free(d_Wz_N);
  fftw_free(d_Wz_N1);
  fftw_free(d_VWx_N);
  fftw_free(d_VWx_N1);
  fftw_free(d_VWy_N);
  fftw_free(d_VWy_N1);


  fftw_free(Vx_N);
  fftw_free(Vy_N);
  fftw_free(Vx_N1);
  fftw_free(Vy_N1);
  fftw_free(PI_N);
  fftw_free(PI_N1);
  fftw_free(Wz_N1);
  fftw_free(Wz_N);
  fftw_free(VWx_N);
  fftw_free(VWx_N1);
  fftw_free(VWy_N);
  fftw_free(VWy_N1);
  fftw_free(Vx_temp);
  fftw_free(Vy_temp);
  
  return 0;
}
//******************************************** FFT FORWARD; taking FFT and killing highest modes ************************************
void fft_forward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V){
  fftw_execute(plan);
  for ( int i=0;i<Nx;i++){
    V[I(i,Ny/2,Ny/2+1)][0] = 0.0;
    V[I(i,Ny/2,Ny/2+1)][1] = 0.0;
  }
  for ( int j=0;j<Ny/2+1;j++){
    V[I(Nx-1,j,Ny/2+1)][0] = 0.0;
    V[I(Nx-1,j,Ny/2+1)][1] = 0.0;
  }
  return ;
}
//************************************** FFT BACKWARD; taking iFFT and multiplying to make it correct *********************************
void fft_backward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V){
  fftw_execute(plan);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      v[I(i,j,Ny)] = v[I(i,j,Ny)]/(double)Nx /(double)Ny;
    }
  }
  return; 
}
//**************************************** Zero out function to zero the highest mode when needed ***************************************
void zero_out( fftw_complex * V,int Nx, int Ny){
   for ( int i=0;i<Nx;i++){
    V[I(i,Ny/2,Ny/2+1)][0] = 0.0;
    V[I(i,Ny/2,Ny/2+1)][1] = 0.0;
  }
  for ( int j=0;j<Ny/2+1;j++){
    V[I(Nx-1,j,Ny/2+1)][0] = 0.0;
    V[I(Nx-1,j,Ny/2+1)][1] = 0.0;
  }
  return; 
}
//**************************************** Computing the energy of the system **********************************************************
void energy_computation(int Nx, int Ny, double* Vx, double*Vy){

  double energy = 0.0;
  for (int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      energy = 0.5 * (Vx[I(i,j,Ny)]*Vx[I(i,j,Ny)] + Vy[I(i,j,Ny)]*Vy[I(i,j,Ny)]);
    }
  }
  
  cout<<"The enegy is: "<<fixed<<energy<<"\n";

}
