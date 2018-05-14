#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>

//Number of fourier modes 
const int Nx = 16;
const int Ny = 16;
// Number of Chebyshev Modes
const int Nz = 8;

// Making Lz to be in [-1,1] results in 
// Lx and Ly to be different than 0 to 2Pi
const double Lx = 2.0*M_PI;
const double Ly = 2.0*M_PI;
double *Kx,*Ky;


using namespace std;
//************************************Indexing ********************************************
int I3(int i, int j, int l,int N){
//indexing is such that I(i,j,l) means 
//i-th element in x
//j-th element in y
//l-th element in z
  return j+ i*(N)+ l*(Nx*N);
}
//******************** Indexing for 2d Fourier case
inline int I(int i, int j,int N){
  return j + N*i;
}
//**************************************Chebyshev Transformer *************************************
void chebyshev_transformer(int N, double * in, double *out, int forward){
  fftw_plan Chebyshev;
  Chebyshev = fftw_plan_r2r_1d(N, in, out, FFTW_REDFT00, FFTW_ESTIMATE);
  
  if (forward != 1){
    in[0] = in[0]*2.0;//2.0*in[0];
    in[N-1] = in[N-1]*2.0;//2*in[N-1];
  }
  
  fftw_execute(Chebyshev);
  
  if( forward ==1){
    for(int i=0;i<N;i++)
      out[i] = out[i]/(double)(N-1);
    out[0] = out[0]/2.0;
    out[N-1] = out[N-1]/2.0;
  }
  if(forward !=1){
    for(int i=0;i<N;i++)
      out[i] = out[i]/2.0;
  }
  fftw_destroy_plan(Chebyshev);
}
//****************************** FFT FORWARD; taking FFT and killing highest modes *****************************
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
//*************************** FFT BACKWARD; taking iFFT and multiplying to make it correct *********************
void fft_backward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V){
  fftw_execute(plan);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      v[I(i,j,Ny)] = v[I(i,j,Ny)]/(double)Nx /(double)Ny;
    }
  }
  return; 
}
//********************** Zero out function to zero the highest mode when needed *********************************
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
/****************************************************************************************************/
int main(){
  // in here we try to do a CFF (Chebyshev-Fourier-Fourier) 
  // on respevtivly z-y-x transform and then get back to real 
  // we first do Chebyshev transform for each mode of X and Y
  // Then we will take Fourier transform on each of z parts
  // Lets start

  ofstream out_start,out_final;
  out_start.open("First_shape.dat");
  out_final.open("Final_shape.dat");

  
  cout<<"Initializing Kx and Ky..."<<endl;
  //************************Initializing Initial Data*************************************************
  Kx = new double[Nx];
  Ky = new double[Ny];
  
  for ( int i=0;i<Nx/2;i++) Kx[i] = 2.0*M_PI*(double)i/Lx;
  for ( int i=-Nx/2;i<0;i++) Kx[i+Nx] = 2.0*M_PI*(double)i/Lx;
  Kx[Nx/2]=0.0;
  
  for ( int i=0;i<Ny/2;i++) Ky[i] = 2.0*M_PI*(double)i/Ly;
  for ( int i=-Ny/2;i<0;i++) Ky[i+Ny] = 2.0*M_PI*(double)i/Ly;
  Ky[Ny/2]=0.0;
  

  cout<<"Allocating Memory...."<<endl;
//*****************************Allocating memory and intializing fftw plans****************************
  fftw_complex *phi, *phi_plane;
  double *d_phi,*d_phi_plane,*temp_vec,*temp_vec2;
  double x,y,z;

  d_phi = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  phi = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);

  d_phi_plane = fftw_alloc_real(sizeof(double)*Nx*Ny);
  phi_plane = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));

  temp_vec = fftw_alloc_real(sizeof(double)*Nz);
  temp_vec2 = fftw_alloc_real(sizeof(double)*Nz);
  
  fftw_plan forward_phi_plane, backward_phi_plane;
  
  forward_phi_plane = fftw_plan_dft_r2c_2d(Nx,Ny,d_phi_plane,phi_plane,FFTW_ESTIMATE);
  backward_phi_plane = fftw_plan_dft_c2r_2d(Nx,Ny,phi_plane,d_phi_plane,FFTW_ESTIMATE);
  

  cout<<"Initializing the vecotr..."<<endl;
  for(int l=0;l<Nz;l++){
    for(int j=0;j<Ny;j++){
      for(int i=0;i<Nx;i++){
	z = cos(M_PI/(double)(Nz-1)*double(l));
	x = Lx /(double)(Nx) *(double)i;
	y = Ly /(double)(Ny) *(double)j;
	d_phi[I3(i,j,l,Ny)] = z*sin(y);
      }
      out_start<<y<<"\t"<<z<<"\t"<<d_phi[I3(0,j,l,Ny)]<<"\n";
    }
    out_start<<"\n";
  }


  cout<<"Doing forward transform ..."<<endl;

  cout<<"Chebyshev transform for each x-i and yj"<<endl;
  //****************************************Chebyshev Transform for each i and j ***************

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      //Copy the z direction into a temporary vector
      for(int l=0;l<Nz;l++)
	temp_vec[l] = d_phi[I3(i,j,l,Ny)];
      //Chebyshev transform in the temp vecotor
      chebyshev_transformer(Nz, temp_vec,temp_vec2,1.0);
      //copy back data into the phi again 
      for(int l=0;l<Nz;l++)
	d_phi[I3(i,j,l,Ny)]=temp_vec2[l];
      //Now we are in Chebyshev Physica Physical Space
    }
  }

  
  cout<<"Fourier-Fourier transformationfor each z....";
  for(int l=0;l<Nz;l++){
    //copy phi for each z into the phi-plane
    // to do the fourier transform
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny;j++){
	d_phi_plane[I(i,j,Ny)] = d_phi[I3(i,j,l,Ny)];
      }
    }
    
    fft_forward(forward_phi_plane,Nx,Ny,d_phi_plane,phi_plane);
    
    //copy data back into phi -> which is into the fourier space
    
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny/2+1;j++){
	phi[I3(i,j,l,Ny/2+1)][0] = phi_plane[I(i,j,Ny/2+1)][0];
	phi[I3(i,j,l,Ny/2+1)][1] = phi_plane[I(i,j,Ny/2+1)][1];
      }
    }
  }  

  //Now we have phi in zChebyshev- xFourier- yFourier Space


  //getting back to physical domain 
  cout<<"Doing backward transform ..."<<endl;
  // We first do the inverse fourier transform on each z levels
  for(int l=0;l<Nz;l++){
    //copy phi for each z into the phi-plane
    // to do the fourier transform
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny/2+1;j++){
	phi_plane[I(i,j,Ny/2+1)][0] = phi[I3(i,j,l,Ny/2+1)][0];
	phi_plane[I(i,j,Ny/2+1)][1] = phi[I3(i,j,l,Ny/2+1)][1];
      }
    }

    fft_backward(backward_phi_plane,Nx,Ny,d_phi_plane,phi_plane);
    
    //copy data back into phi -> which is into the fourier space
    

    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny;j++){
	d_phi[I3(i,j,l,Ny)]=d_phi_plane[I(i,j,Ny)];
      }
    }
  }
  
  //Now we should do the inverse of Chebyshev transform :D
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      //Copy the z direction into a temporary vector
      for(int l=0;l<Nz;l++)
	temp_vec[l] = d_phi[I3(i,j,l,Ny)];
      //Inverse Chebyshev transform in the temp vecotor
      chebyshev_transformer(Nz, temp_vec,temp_vec2,-1.0);
      //copy back data into the phi again 
      for(int l=0;l<Nz;l++)
	d_phi[I3(i,j,l,Ny)]=temp_vec2[l];
      //Now we are in Physical Physical Physical Space
    }
  }
  // Now we are safe :D Lets now check :D

  for(int l=0;l<Nz;l++){
    for(int j=0;j<Ny;j++){
      for(int i=0;i<Nx;i++){
	z = cos(M_PI/(double)(Nz-1)*double(l));
	x = Lx /(double)(Nx) *(double)i;
	y = Ly /(double)(Ny) *(double)j;
	//	d_phi[I3(i,j,l,Ny)] = z*sin(y);
      }
      out_final<<"\t"<<y<<"\t"<<z<<"\t"<<d_phi[I3(0,j,l,Ny)]<<"\n";
    }
    out_final<<"\n";
  }
  
 fftw_free(d_phi);
 fftw_free(d_phi_plane);
 fftw_free(temp_vec);
 fftw_free(temp_vec2);
 fftw_free(phi);
 fftw_free(phi_plane);
 fftw_destroy_plan(forward_phi_plane);
 fftw_destroy_plan(backward_phi_plane);
 
 return 0;



}
/************************************************/
