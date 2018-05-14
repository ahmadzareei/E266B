#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>

using namespace std;

/**********************************************************************************************************************/
inline int I(int i, int j,int Ny){
  return j + Ny*i;
}

void AdamsBashforthAdvection2D(fftw_complex * in, double * double_in, const int Nx, const int Ny,\
			      const double dt, const double T, double * Kx, double * Ky, \
			       fftw_plan fft_forward_r2c, fftw_plan fft_backward_c2r);

void  Kill_Highest_Term(fftw_complex* in0, int const Nx, int const Ny);
void F_Advection(double *VdV0,fftw_complex *in0, double * double_in0,double * Kx, double * Ky,const int Nx, const int Ny, \
		 fftw_plan fft_forward_r2c, fftw_plan fft_backward_c2r); 
void Set_Equal(double * in_left, double *in_right,const int Nx,const int Ny);
void AdamsBashforthAdvection2DStep(double *Vn,double* VdV0,double *VdV1,fftw_complex* in0, double* double_in0,\
				   double *Kx, double *Ky,const int Nx,const int Ny,fftw_plan fft_forward_r2c, fftw_plan fft_backward_c2r,double dt);
void Rescale(double * input, const int Nx, const int Ny);
/**********************************************************************************************************************/

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
  const double dt = 0.0001;
  const double T = 0.5;


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
      input_data[I(i,j,Ny)] = sin(Lx*(double)i/(double)Nx);// * cos(Ly*(double)j/(double)Ny);
      out1<<Lx*(double)(i)/(double)Nx<<"\t"<<Ly*(double)j/(double)Ny<<"\t"<<input_data[I(i,j,Ny)]<<"\n";
    }
    out1<<"\n";
  }


  
  AdamsBashforthAdvection2D(fft_input_data,input_data,Nx,Ny,dt,T,Kx,Ky, fft_forward_r2c,fft_backward_c2r);


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
/****************************************************************************************************/
void AdamsBashforthAdvection2DStep(double *Vn,double* VdV0,double *VdV1,fftw_complex* in0, double* double_in0,double *Kx, double *Ky,const int Nx,const int Ny,fftw_plan fft_forward_r2c, fftw_plan fft_backward_c2r,double dt){

  // finding New value of Vn
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny;j++){
      Vn[I(i,j,Ny)] = Vn[I(i,j,Ny)] + dt/2.0*(3*VdV1[I(i,j,Ny)]-VdV0[I(i,j,Ny)]);
    }
  }
  
  // putting VdV1 -> VdV0
  Set_Equal(VdV0,VdV1,Nx,Ny);
  
  //putting Vn -> double_in0
  Set_Equal(double_in0,Vn,Nx,Ny);
  
  F_Advection(VdV1,in0,double_in0,Kx,Ky,Nx,Ny,fft_forward_r2c,fft_backward_c2r);
  
}
/*********************************************************************************************************/
//Adams-Bashforth Equation for the Nonliner Advection Part ( Vt = VdV/dx)
// in = input, out = output, Nx = x-domain size, Ny = y-domain size,
// dt time step, T= final time
// Kx = frequency coefficient in x direction
// Ky = frequency coefficient in y direction
void AdamsBashforthAdvection2D(fftw_complex * in0, double *double_in0 , const int Nx, const int Ny,\
			       const double dt, const double T, double * Kx, double * Ky,\
			       fftw_plan fft_forward_r2c, fftw_plan fft_backward_c2r){

  double time;
  
  // Store VdV/dx at time n-1
  double *VdV0;
  VdV0 = fftw_alloc_real(sizeof(double)*Nx*Ny);

  // Store VdV/dx at time n
  double *VdV1;
  VdV1 = fftw_alloc_real(sizeof(double)*Nx*Ny);
  
  // Store the last solution
  double *Vn;
  Vn = fftw_alloc_real(sizeof(double)*Nx*Ny);
  // setting the last solution
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny;j++){
      Vn[I(i,j,Ny)] = double_in0[I(i,j,Ny)];
    }
  }

  F_Advection(VdV0,in0,double_in0,Kx,Ky,Nx,Ny,fft_forward_r2c,fft_backward_c2r);

  // This part should be Runge-Kutta 2nd order for the first step
  //somehow we get the next solution and for that we compute the term VdV1  
  //with euler interpolation we have
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny;j++){
      Vn[I(i,j,Ny)] = Vn[I(i,j,Ny)] + dt*VdV0[I(i,j,Ny)];
    }
  }

  //Set equal for computation that we will have 
  Set_Equal(double_in0,Vn,Nx,Ny);
  F_Advection(VdV1,in0,double_in0,Kx,Ky,Nx,Ny,fft_forward_r2c,fft_backward_c2r);

  //while loop to do a step in Adams-Bashforth Method
  time = 0.0;
  while (time<T){
    AdamsBashforthAdvection2DStep(Vn,VdV0,VdV1,in0,double_in0,Kx,Ky,Nx,Ny,fft_forward_r2c,fft_backward_c2r,dt);
    time += dt;
  }
  
  Set_Equal(double_in0, Vn, Nx,Ny);
}
/*****************************************************************************************************/
void  Kill_Highest_Term(fftw_complex* in0, int const Nx, int const Ny){
  
  for ( int j=0;j<Ny/2+1;j++){
    in0[I(Nx/2,j,Ny/2+1)][0] = 0.0;
    in0[I(Nx/2,j,Ny/2+1)][1] = 0.0;
  }
  for ( int i=0;i<Nx;i++){
    in0[I(i,Ny/2,Ny/2+1)][0] = 0.0;
    in0[I(i,Ny/2,Ny/2+1)][1] = 0.0;    
  }
}

/****************************************************************************************************/
void F_Advection(double *VdV0,fftw_complex *in0, double * double_in0,double * Kx, double *Ky,const int Nx, const int Ny,fftw_plan fft_forward_r2c, fftw_plan fft_backward_c2r){ 

  double temp;
  
  Set_Equal(VdV0, double_in0,Nx,Ny);

  fftw_execute(fft_forward_r2c);  
  Kill_Highest_Term(in0,Nx,Ny);

  //taking derivative of input
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny/2+1;j++){
      temp = in0[I(i,j,Ny/2+1)][0];
      in0[I(i,j,Ny/2+1)][0] = -in0[I(i,j,Ny/2+1)][1]*Kx[i];
      in0[I(i,j,Ny/2+1)][1] = temp*Kx[i];
    }
  }

  fftw_execute(fft_backward_c2r);
  Rescale(double_in0,Nx,Ny);

  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny;j++){
      VdV0[I(i,j,Ny)] =  VdV0[I(i,j,Ny)]*double_in0[I(i,j,Ny)];
    }
  }  

}
/*************************************************************************************************/
void Set_Equal(double * in_left, double *in_right,const int Nx,const int Ny){
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny;j++){
      in_left[I(i,j,Ny)] = in_right[I(i,j,Ny)];
    }
  }  
}
/***********************************************************************************************/
void Rescale(double * input, const int Nx, const int Ny){
  for ( int i=0;i<Nx;i++){
    for ( int j=0;j<Ny;j++){
      input[I(i,j,Ny)] = input[I(i,j,Ny)]/(double)Nx/(double)Ny;
    }
  }
}
