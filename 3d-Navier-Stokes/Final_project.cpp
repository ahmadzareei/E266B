#define  ARMA_DONT_USE_WRAPPER
#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>
#include<armadillo>

using namespace std;
using namespace arma;
//****************************************************************************************************
//****************************************************************************************************
//****************************Very first initial Condition********************************************
//****************************************************************************************************
//****************************************************************************************************
//Number of Fourier modes
const int Nx = 16;
const int Ny = 16;
// Number of Chebyshev Modes
const int Nz =17;


// Lz is in [-1,1]
// Making Lz to be in [-1,1] results in 
// Lx and Ly to be different than [0,2Pi]
const double Lx = M_PI/8.0;
const double Ly = M_PI/8.0;
double *Kx,*Ky;
const double dt = 0.001;
const double T = 1000*dt;
double nu = 0.0;

//****************************************************************************************************
//****************************************************************************************************
//**********************************Indexing Functions************************************************
//****************************************************************************************************
//****************************************************************************************************
int I3(int i, int j, int l,int N){
//indexing is such that I(i,j,l) means 
//i-th element in x
//j-th element in y
//l-th element in z
  return j+ i*(N)+ l*(Nx*N);
}
//******************** Indexing for 2d Fourier case*********************
inline int I(int i, int j,int N){
  return j + N*i;
}

//***************************************************************************************************
//***************************************************************************************************
//**************************************Chebyshev Transformer ***************************************
//***************************************************************************************************
//***************************************************************************************************
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
//*****************************************************************************************************
//*****************************************************************************************************
//********************* FFT FORWARD; taking FFT and killing highest modes *****************************
//*****************************************************************************************************
//*****************************************************************************************************
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
//***************** FFT BACKWARD; taking iFFT and multiplying to make it correct *********************
void fft_backward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V){
  fftw_execute(plan);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      v[I(i,j,Ny)] = v[I(i,j,Ny)]/(double)Nx /(double)Ny;
    }
  }
  return; 
}
//*************** Zero out function to zero the highest mode when needed ******************************
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
//****************************************************************************************************
//****************************************************************************************************
//***************************Taking Chebyshev Derivative**********************************************
//****************************************************************************************************
//****************************************************************************************************
void chebyshev_derivative(int N, double * in, double * out){
  // in and out should be different
  // takes derivative of in, supposing in[i] is coefficient of 
  // T_i(x) where , T_i is the ith chebyshev function
  out[N-1]= 0.0;
  out[N-2] = 2.0*(double)(N-1)*in[N-1];
  
  for(int k=N-3;k>0;k-=1){
    out[k] = 2*(double)(k+1)*in[k+1] + out[k+2];
  }
  out[0] = in[1] + out[2]/2.0;
}
//****************************************************************************************************
//****************************************************************************************************
//******************************* Paper cutter Method ************************************************
//****************************************************************************************************
//****************************************************************************************************
void papper_cutter(int N, double * A, double *X, double *Y){
  //This procedure will solve AX=Y using paper cutter Method
  // supposing A and Y are known, we will find X
  // There is also an assumption on A
  // A has the form D^2 (in chebyshev space) - (m^2 + n^2)*I where m and n are not zero
  // where I is the identity matrix 
  // last two rows of A are removed and [1 1 1 ...  1 1]
  // and [1 -1 1 -1 ....] are substitued there
  double *G1,*G2,*G3,*c1,*c2,*r1,*r2;
  double temp,R1,R2;
  double A11,A12,A21,A22;
  G1 = fftw_alloc_real(sizeof(double)*(N-2));
  G2 = fftw_alloc_real(sizeof(double)*(N-2));
  G3 = fftw_alloc_real(sizeof(double)*(N-2));
  c1 = fftw_alloc_real(sizeof(double)*(N-2));
  c2 = fftw_alloc_real(sizeof(double)*(N-2));
  r1 = fftw_alloc_real(sizeof(double)*(N-2));
  r2 = fftw_alloc_real(sizeof(double)*(N-2));

  for(int i=0;i<N-2;i++){
    c1[i] = A[I(i,N-2,N)];
    c2[i] = A[I(i,N-1,N)];
    r1[i] = A[I(N-2,i,N)];
    r2[i] = A[I(N-1,i,N)];
  }



  //Building G1
  G1[N-3] = c1[N-3]/A[I(N-3,N-3,N)];
  G1[N-4] = c1[N-4]/A[I(N-4,N-4,N)];
  for(int i = N-5;i>=0;i--){
    temp = 0.0;
    for(int j = i+2;j<=N-3;j=j+2){
      temp = temp + A[I(i,j,N)]*G1[(j)];
    }
    G1[i] = (c1[i]-temp)/A[I(i,i,N)];
  }

  //Building G2
  G2[N-3] = c2[N-3]/A[I(N-3,N-3,N)];
  G2[N-4] = c2[N-4]/A[I(N-4,N-4,N)];
  for(int i = N-5;i>=0;i--){
    temp = 0.0;
    for(int j = i+2;j<=N-3;j=j+2){
      temp = temp + A[I(i,j,N)]*G2[(j)];
    }
    G2[i] = (c2[i]-temp)/A[I(i,i,N)];
  }

  //Building G3
  G3[N-3] = Y[N-3]/A[I(N-3,N-3,N)];
  G3[N-4] = Y[N-4]/A[I(N-4,N-4,N)];
  for(int i = N-5;i>=0;i--){
    temp = 0.0;
    for(int j = i+2;j<=N-3;j=j+2){
      temp = temp + A[I(i,j,N)]*G3[(j)];
    }
    G3[i] = (Y[i]-temp)/A[I(i,i,N)];
  }

  //Building A11 up to A22
  temp = 0;
  for(int i=0;i<N-2;i++){
    temp = temp  + G1[i] * r1[i];
  }
  A11 = A[I(N-2,N-2,N)] - temp;

  temp = 0;
  for(int i=0;i<N-2;i++){
    temp = temp  + G2[i] * r1[i];
  }
  A12 = A[I(N-2,N-1,N)] - temp;
 
  temp = 0;
  for(int i=0;i<N-2;i++){
    temp = temp  + G1[i] * r2[i];
  }
  A21 = A[I(N-1,N-2,N)] - temp;
 
  temp = 0;
  for(int i=0;i<N-2;i++){
    temp = temp  + G2[i] * r2[i];
  }
  A22 = A[I(N-1,N-1,N)] - temp;
  
  //Building the RHS, vector of [R1;R2]
  temp = 0; 
  for(int i=0;i<N-2;i++){
    temp = temp + r1[i]*G3[i];
  }
  R1 = Y[N-2] - temp;

  temp = 0; 
  for(int i=0;i<N-2;i++){
    temp = temp + r2[i]*G3[i];
  }
  R2 = Y[N-1] - temp;

  //Now we have two sets of equations as 
  // A11 *X[N-2] + A12 * X[N-1] = R1
  // A21 *X[N-2] + A22 * X[N-1] = R2
  
  X[N-2] = (R1*A22 - R2*A12)/(A22*A11-A12*A21);
  X[N-1] = (R2*A11 - R1*A21)/(A22*A11-A12*A21);
  
  for(int i = N-3;i>=0;i--){
    X[i] = -X[N-2]*G1[i] - X[N-1]*G2[i] + G3[i];
  }
  fftw_free(G1);
  fftw_free(G2) ;
  fftw_free(G3);
  fftw_free(c1);
  fftw_free(c2);
  fftw_free(r1);
  fftw_free(r2);
  return;
}

//****************************************************************************************************
//****************************************************************************************************
//*********************************CFF Transformer Forward *******************************************
//****************************************************************************************************
//****************************************************************************************************
void CFF_transformer_forward(double * d_phi2, fftw_complex *phi){

  //Allocating Memory needed
  double * d_phi_plane;
  double *temp_vec,*temp_vec2;
  fftw_complex * phi_plane;
  
  temp_vec = fftw_alloc_real(sizeof(double)*Nz);
  temp_vec2 = fftw_alloc_real(sizeof(double)*Nz);  
  d_phi_plane = fftw_alloc_real(sizeof(double)*Nx*Ny);
  phi_plane = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));

  double * d_phi;
  d_phi = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	d_phi[I3(i,j,l,Ny)] = d_phi2[I3(i,j,l,Ny)];
      }
    }
  }

  fftw_plan forward_phi_plane, backward_phi_plane;
  
  forward_phi_plane = fftw_plan_dft_r2c_2d(Nx,Ny,d_phi_plane,phi_plane,FFTW_ESTIMATE);
  backward_phi_plane = fftw_plan_dft_c2r_2d(Nx,Ny,phi_plane,d_phi_plane,FFTW_ESTIMATE);
  
  
  //Chebyshev transformation on each i and j
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

  
  //cout<<"Fourier-Fourier transformation or each z....";
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
  fftw_free(phi_plane);
  fftw_free(d_phi_plane);
  fftw_free(temp_vec);
  fftw_free(temp_vec2);
  fftw_destroy_plan(forward_phi_plane);
  fftw_destroy_plan(backward_phi_plane);


}
//****************************************************************************************************
//****************************************************************************************************
//***********************************CFF Transformer Backward*****************************************
//****************************************************************************************************
//****************************************************************************************************
void CFF_transformer_backward(double * d_phi, fftw_complex *phi){
  
  //Allocating Memory needed
  double * d_phi_plane;
  double *temp_vec,*temp_vec2;
  fftw_complex * phi_plane;
  
  temp_vec = fftw_alloc_real(sizeof(double)*Nz);
  temp_vec2 = fftw_alloc_real(sizeof(double)*Nz);  
  d_phi_plane = fftw_alloc_real(sizeof(double)*Nx*Ny);
  phi_plane = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));

  fftw_plan forward_phi_plane, backward_phi_plane;
  
  forward_phi_plane = fftw_plan_dft_r2c_2d(Nx,Ny,d_phi_plane,phi_plane,FFTW_ESTIMATE);
  backward_phi_plane = fftw_plan_dft_c2r_2d(Nx,Ny,phi_plane,d_phi_plane,FFTW_ESTIMATE);

  
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
  
  //Now we should do the inverse of Chebyshev transform 
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

  
  fftw_free(phi_plane);
  fftw_free(d_phi_plane);
  fftw_free(temp_vec);
  fftw_free(temp_vec2);
  fftw_destroy_plan(forward_phi_plane);
  fftw_destroy_plan(backward_phi_plane);

}
//****************************************************************************************************
//****************************************************************************************************
//************************************ Return e_i ****************************************************
//****************************************************************************************************
//****************************************************************************************************
void Ai(int N, double *A, int index){
  for(int i=0;i<N;i++){
    A[i] = 0.0;
  }
  A[index]= 1.0;
}

//****************************************************************************************************
//****************************************************************************************************
//************************************ Building D2 ***************************************************
//****************************************************************************************************
//****************************************************************************************************
void  build_D2(int N, double *D2){
  // D2 is a matrix, that acts on a vector in Chebyshev Space
  // and gives back the second derivative of it
  // last two rows are substitued with [ 1 1 1 ...1]
  // and [1 -1 1 -1 1 ... ]
  // Theta = D2 -(kx^2 + ky^2)I
  // Building Theta Matrix => This will be D*D matrix
  // With last to rows being [1 1 1 1... 1]
  // and [1 -1 1 -1 ......]
  double *A,*B;
  A = fftw_alloc_real(sizeof(double)*N);
  B = fftw_alloc_real(sizeof(double)*N);
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++)
      D2[I(i,j,N)]=0.0;
    A[i] = 0.0;
    B[i] = 0.0;
  }
  for(int i=0;i<N;i++){
    Ai(N,A,i);    
    chebyshev_derivative(N,A,B);
    chebyshev_derivative(N,B,A);
    
    for(int j=0;j<N;j++){
      D2[I(j,i,N)] = A[j];
    }
  }

  D2[I(N-2,0,N)] = 1.0;
  D2[I(N-1,0,N)] = 1.0;
  
  for(int i=1;i<N;i++){
    D2[I(N-2,i,N)] = 1.0;
    D2[I(N-1,i,N)] = -1.0*D2[I(N-1,i-1,N)];
  }
}
//****************************************************************************************************
//****************************************************************************************************
//***************************** Building G1,G2,G7,G8 *************************************************
//****************************************************************************************************
//****************************************************************************************************
void  build_G1278(double *D2,double *G1,double *G2,double *G7,double *G8){

  double * temp_v1, *temp_v2;
  temp_v1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_v2 = fftw_alloc_real(sizeof(double)*Nz);
  for(int i=0;i<Nz;i++){
    temp_v1[i] = 0.0;
    temp_v2[i] = 0.0;
  }

  //remove th highest mode to be zero
  //this will cause problems!!
  //Building G7  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      if(i!=0 | j!=0){
	//first we write the D2 on the diagonal
	for(int l=0;l<Nz-2;l++)
	  D2[I(l,l,Nz)] = -Kx[i]*Kx[i] - Ky[j]*Ky[j];

	// Building Tm-1(z) in temp_v1
	Ai(Nz,temp_v1,Nz-2);
	papper_cutter(Nz, D2,temp_v2,temp_v1); 
	for(int l=0;l<Nz;l++)
	  G7[I3(i,j,l,Ny/2+1)] = temp_v2[l];

	//Building dTm-1(z)/dz with last two element zero
	// now temp_v1 = e_{N-2}, we first take the derivative
	chebyshev_derivative(Nz,temp_v1,temp_v2);
	temp_v2[Nz-1]=0.0;
	temp_v2[Nz-2] = 0.0;
	
	/*
	for(int l=0;l<Nz;l++)
	  cout<<temp_v2[l]<<endl;
	*/

	papper_cutter(Nz,D2,temp_v1,temp_v2);
	for(int l=0;l<Nz;l++)
	  G1[I3(i,j,l,Ny/2+1)] = temp_v1[l];
	


	// Building Tm(z) in temp_v1
	Ai(Nz,temp_v1,Nz-1);
	papper_cutter(Nz,D2,temp_v2,temp_v1); 
	for(int l=0;l<Nz;l++)
	  G8[I3(i,j,l,Ny/2+1)] = temp_v2[l];
	
	//Building dTm(z)/dz with last two element zero
	// now temp_v1 = e_{N-1}, we first take the derivative
	chebyshev_derivative(Nz,temp_v1,temp_v2);
	temp_v2[Nz-1]=0.0;
	temp_v2[Nz-2] = 0.0;
	papper_cutter(Nz,D2,temp_v1,temp_v2);
	for(int l=0;l<Nz;l++)
	  G2[I3(i,j,l,Ny/2+1)] = temp_v1[l];
      }
    }
  }
}
//****************************************************************************************************
//****************************************************************************************************
//************************** Taking Dz of the whole Mataix********************************************
//****************************************************************************************************
// This function do the chebyshev derivative on each column of the matrix in CFF space ***************
//****************************************************************************************************
void Dz(fftw_complex *in, fftw_complex *out){
  // both in and out are in complex space
  double *temp_vec1,*temp_vec2;
  temp_vec1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_vec2 = fftw_alloc_real(sizeof(double)*Nz);  
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int mod =0; mod <2;mod++){
	//Copy the z direction into a temporary vector
	for(int l=0;l<Nz;l++)
	  temp_vec1[l] = in[I3(i,j,l,Ny/2+1)][mod];
	//taking chebyshev derivative
	chebyshev_derivative(Nz, temp_vec1,temp_vec2);
	//compy bach to the output 
	for(int l=0;l<Nz;l++)
	  out[I3(i,j,l,Ny/2+1)][mod]=temp_vec2[l];
      }
    }
  }

}
//****************************************************************************************************
//****************************************************************************************************
//************************** Taking Dz of the whole real Mataix***************************************
//****************************************************************************************************
// This function do the chebyshev derivative on each column of the matrix in CFF space ***************
//****************************************************************************************************
void Dz_real(double *in, double *out){
  // both in and out are in complex space
  double *temp_vec1,*temp_vec2;
  temp_vec1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_vec2 = fftw_alloc_real(sizeof(double)*Nz);  
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
	//Copy the z direction into a temporary vector
	for(int l=0;l<Nz;l++)
	  temp_vec1[l] = in[I3(i,j,l,Ny/2+1)];
	//taking chebyshev derivative
	chebyshev_derivative(Nz, temp_vec1,temp_vec2);
	//compy bach to the output 
	for(int l=0;l<Nz;l++)
	  out[I3(i,j,l,Ny/2+1)]=temp_vec2[l];
    }
  }

}

//****************************************************************************************************
//****************************************************************************************************
//************************** Taking laplacian of the whole Mataix********************************************
//****************************************************************************************************
// This function do the chebyshev derivative on each column of the matrix in CFF space ***************
//****************************************************************************************************
void laplacian(fftw_complex *in, fftw_complex *out){
  // both in and out are in complex space
  double *temp_vec1,*temp_vec2;
  temp_vec1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_vec2 = fftw_alloc_real(sizeof(double)*Nz);  
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int mod =0; mod <2;mod++){
	//Copy the z direction into a temporary vector
	for(int l=0;l<Nz;l++)
	  temp_vec1[l] = in[I3(i,j,l,Ny/2+1)][mod];
	//taking chebyshev derivative
	chebyshev_derivative(Nz, temp_vec1,temp_vec2);
	//taking chebyshev derivative
	chebyshev_derivative(Nz, temp_vec2,temp_vec1);
	//compy bach to the output 
	for(int l=0;l<Nz;l++)
	  out[I3(i,j,l,Ny/2+1)][mod]= temp_vec1[l] - ( Kx[i]*Kx[i] + Ky[j]*Ky[j] )* in[I3(i,j,l,Ny/2+1)][mod];
      }
    }
  }
}

//****************************************************************************************************
//****************************************************************************************************
//********************** Calculating Vorticity from Velocity *****************************************
//****************************************************************************************************
//****************************************************************************************************
void omega_calculator(double *d_Wx,double *d_Wy,double *d_Wz,double *d_temp1,double *d_temp2,fftw_complex *Vx,fftw_complex *Vy,fftw_complex *Vz,fftw_complex *temp1,fftw_complex *temp2){
  
  // Computing d_Wx
  // d_Wx = dVz/dy - dVy/dz
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	temp1[I3(i,j,l,Ny/2+1)][0] = -Ky[j] * Vz[I3(i,j,l,Ny/2+1)][1];
  	temp1[I3(i,j,l,Ny/2+1)][1] =  Ky[j] * Vz[I3(i,j,l,Ny/2+1)][0];
      }
    }
  }
  Dz(Vy,temp2);
  CFF_transformer_backward(d_temp1,temp1);
  CFF_transformer_backward(d_temp2,temp2);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	d_Wx[I3(i,j,l,Ny)] = d_temp1[I3(i,j,l,Ny)]-d_temp2[I3(i,j,l,Ny)];
      }
    }
  }

  // Computing d_Wy
  // d_Wy = dVx/dz - dVz/dx
  Dz(Vx,temp1);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	temp2[I3(i,j,l,Ny/2+1)][0] = -Kx[i] * Vz[I3(i,j,l,Ny/2+1)][1];
  	temp2[I3(i,j,l,Ny/2+1)][1] =  Kx[i] * Vz[I3(i,j,l,Ny/2+1)][0];
      }
    }
  }
  CFF_transformer_backward(d_temp1,temp1);
  CFF_transformer_backward(d_temp2,temp2);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	d_Wy[I3(i,j,l,Ny)] = d_temp1[I3(i,j,l,Ny)] - d_temp2[I3(i,j,l,Ny)];
      }
    }
  }
  //computing d_Wz
  // d_Wz = dVy/dx - dVx/dy
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	temp1[I3(i,j,l,Ny/2+1)][0] = -Kx[i] * Vy[I3(i,j,l,Ny/2+1)][1];
	temp1[I3(i,j,l,Ny/2+1)][1] =  Kx[i] * Vy[I3(i,j,l,Ny/2+1)][0];

	temp2[I3(i,j,l,Ny/2+1)][0] = -Ky[j] * Vx[I3(i,j,l,Ny/2+1)][1];
	temp2[I3(i,j,l,Ny/2+1)][1] =  Ky[j] * Vx[I3(i,j,l,Ny/2+1)][0];
      }
    }
  }
  CFF_transformer_backward(d_temp1,temp1);
  CFF_transformer_backward(d_temp2,temp2);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	d_Wz[I3(i,j,l,Ny)] = d_temp1[I3(i,j,l,Ny)] - d_temp2[I3(i,j,l,Ny)];
      }
    }
  } 
}
//****************************************************************************************************
//****************************************************************************************************
//*********************** Taking cross product of VxW ************************************************
//****************************************************************************************************
//****************************************************************************************************
void cross_product(double *d_Vx,double *d_Vy,double *d_Vz, double *d_Wx, double *d_Wy, double *d_Wz, double *d_VWx, double *d_VWy, double *d_VWz,fftw_complex  *VWx, fftw_complex *VWy,fftw_complex * VWz){
// We take inputs of (d_Vx,d_Vy,d_Vz) x (d_Wx, d_Wy, d_Wz)
// put the result into (d_VWx, d_VWy, d_VWz)
// taking the CFF transform and putting the result back in (VWx, VWy, VWz)
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	d_VWx[I3(i,j,l,Ny)] = d_Vy[I3(i,j,l,Ny)] * d_Wz[I3(i,j,l,Ny)] - d_Vz[I3(i,j,l,Ny)] * d_Wy[I3(i,j,l,Ny)];
	d_VWy[I3(i,j,l,Ny)] = d_Vz[I3(i,j,l,Ny)] * d_Wx[I3(i,j,l,Ny)] - d_Vx[I3(i,j,l,Ny)] * d_Wz[I3(i,j,l,Ny)];
	d_VWz[I3(i,j,l,Ny)] = d_Vx[I3(i,j,l,Ny)] * d_Wy[I3(i,j,l,Ny)] - d_Vy[I3(i,j,l,Ny)] * d_Wx[I3(i,j,l,Ny)];
      }
    }
  }
  CFF_transformer_forward(d_VWx, VWx);
  CFF_transformer_forward(d_VWy, VWy);
  CFF_transformer_forward(d_VWz, VWz);
}

//****************************************************************************************************
//****************************************************************************************************
//***************************************** Pi^ builder **********************************************
//****************************************************************************************************
//****************************************************************************************************
void  build_pi_hat(fftw_complex *Vx, fftw_complex *Vy, fftw_complex *Vz, double *D2, fftw_complex *PI, fftw_complex *temp1){
  // basically if set the last two rows of [V^N+1/3+ nu*dt/2 * laplacian(V^N)] to zero
  // and multiply with theta^-1 we will have Phi^ !!!
  // Lets start
  double * temp_v1, *temp_v2;
  temp_v1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_v2 = fftw_alloc_real(sizeof(double)*Nz);
  for(int i=0;i<Nz;i++){
    temp_v1[i] = 0.0;
    temp_v2[i] = 0.0;
  }
  
  //taking the d/dz of Vz^N+1/3+ nu*dt/2 * laplacian(Vz^N)
  Dz(Vz,temp1);
  

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      if(i!=0 & j!=0){
	//first we write the D2 on the diagonal
	for(int l=0;l<Nz-2;l++)
	  D2[I(l,l,Nz)] = -Kx[i]*Kx[i] - Ky[j]*Ky[j];
	// Now we have our D2 for the papper cutter method
	for(int l=0;l<Nz;l++){
	  temp_v1[l] = -Kx[i]*Vx[I3(i,j,l,Ny/2+1)][1] - Ky[j]*Vy[I3(i,j,l,Ny/2+1)][1] + temp1[I3(i,j,l,Ny/2+1)][0];
	}
	temp_v1[Nz-2]=0.0;
	temp_v1[Nz-1]=0.0;
	papper_cutter(Nz,D2,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  PI[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
	}
	for(int l=0;l<Nz;l++){
	  temp_v1[l] = Kx[i]*Vx[I3(i,j,l,Ny/2+1)][0] + Ky[j]*Vy[I3(i,j,l,Ny/2+1)][0] + temp1[I3(i,j,l,Ny/2+1)][1];
	}
	temp_v1[Nz-2]=0.0;
	temp_v1[Nz-1]=0.0;
	papper_cutter(Nz,D2,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  PI[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
      }
    }
  }
}
//****************************************************************************************************
//****************************************************************************************************
//****************************************************************************************************
//****************************** Computing Phi^-1 gradG_i ********************************************
//****************************************************************************************************
//****************************************************************************************************
void  compute_phi_inv_grad_G(fftw_complex *phi_inv_grad_G1x,fftw_complex * phi_inv_grad_G2x,fftw_complex * phi_inv_grad_G7x, fftw_complex *phi_inv_grad_G8x,fftw_complex *phi_inv_grad_G1y,fftw_complex * phi_inv_grad_G2y,fftw_complex * phi_inv_grad_G7y, fftw_complex *phi_inv_grad_G8y,fftw_complex *phi_inv_grad_G1z,fftw_complex * phi_inv_grad_G2z,fftw_complex * phi_inv_grad_G7z, fftw_complex *phi_inv_grad_G8z,double * D2,double *d_G1,double * d_G2,double * d_G7,double * d_G8,double * d_gradz_G1,double *d_gradz_G2,double *d_gradz_G7, double *d_gradz_G8){
  double * temp_v1, *temp_v2;
  temp_v1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_v2 = fftw_alloc_real(sizeof(double)*Nz);
  for(int i=0;i<Nz;i++){
    temp_v1[i] = 0.0;
    temp_v2[i] = 0.0;
  }
  double *phii;
  phii = fftw_alloc_real(sizeof(double)*Nz*Nz);
  for(int i=0;i<Nz;i++){
    for(int j=0;j<Nz;j++){
      phii[I(i,j,Nz)] = -nu*dt/2.0* D2[I(i,j,Nz)];
    }
  }
  phii[I(Nz-2,0,Nz)] = 1.0;
  phii[I(Nz-1,0,Nz)] = 1.0;
  
  for(int i=1;i<Nz;i++){
    phii[I(Nz-2,i,Nz)] = 1.0;
    phii[I(Nz-1,i,Nz)] = -1.0*phii[I(Nz-1,i-1,Nz)];
  }
  // since G1, G2, G7 and G8 are real, then gradient in x direction, they will have imaginery part
  // so all real part of these values in x and y direction is zero 
  // and also , since gradient in z direction is real, then the imaginary part of gradient in z
  // for G1, G2, G7 and G8 will be zero
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	phi_inv_grad_G1x[I3(i,j,l,Ny/2+1)][0] =0.0;
	phi_inv_grad_G1y[I3(i,j,l,Ny/2+1)][0] =0.0;

	phi_inv_grad_G2x[I3(i,j,l,Ny/2+1)][0] =0.0;
	phi_inv_grad_G2y[I3(i,j,l,Ny/2+1)][0] =0.0;
  
	phi_inv_grad_G7x[I3(i,j,l,Ny/2+1)][0] =0.0;
	phi_inv_grad_G7y[I3(i,j,l,Ny/2+1)][0] =0.0;

	phi_inv_grad_G8x[I3(i,j,l,Ny/2+1)][0] =0.0;
	phi_inv_grad_G8y[I3(i,j,l,Ny/2+1)][0] =0.0;

	phi_inv_grad_G1z[I3(i,j,l,Ny/2+1)][1] =0.0;
	phi_inv_grad_G2z[I3(i,j,l,Ny/2+1)][1] =0.0;
	phi_inv_grad_G7z[I3(i,j,l,Ny/2+1)][1] =0.0;
	phi_inv_grad_G8z[I3(i,j,l,Ny/2+1)][1] =0.0;
      }
    }
  }
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
        // building phi
        if(i !=0 | j !=0){
        for(int l=0;l<Nz-2;l++){
	  phii[I(l,l,Nz)] = 1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]);
	}
	
	/*
	//***********************************
	// For debugging reasons !!!
	if(i ==2 & j==1){
	  cout<<"i =2 and j=1, the phii matrix is:"<<endl;
	  for(int l=0;l<Nz;l++){
	    for(int ll=0;ll<Nz;ll++){
	      cout<<phii[I(l,ll,Nz)]<<"\t";}
	    cout<<"\n";
	  }
	}
	///************************************************
	*/
	
	// computing phi_inv_grad_G1
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Kx[i] * d_G1[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G1x[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
	
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Ky[j] * d_G1[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G1y[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
	
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = d_gradz_G1[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G1z[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
	}
	
	
	// computing phi_inv_grad_G2
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Kx[i] * d_G2[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	phi_inv_grad_G2x[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
	
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Ky[j] * d_G2[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G2y[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
	
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = d_gradz_G2[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G2z[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
	}
	
	
	// computing phi_inv_grad_G7
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Kx[i] * d_G7[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G7x[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}

	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Ky[j] * d_G7[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G7y[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
	
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = d_gradz_G7[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G7z[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
	}
	
	
	// computing phi_inv_grad_G8
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Kx[i] * d_G8[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G8x[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
	
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = Ky[j] * d_G8[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G8y[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
	}
	
	for(int l=0;l<Nz;l++)
	  temp_v1[l] = d_gradz_G8[I3(i,j,l,Ny/2+1)];
	temp_v1[Nz-1] = 0.0;
	temp_v1[Nz-2] = 0.0;
	papper_cutter(Nz, phii,temp_v2,temp_v1);
	for(int l=0;l<Nz;l++){
	  phi_inv_grad_G8z[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
       }
      }
      
    }
  }
  
}

//****************************************************************************************************
//****************************************************************************************************
//****************************************************************************************************
//********************Building RHS of div.V^N+1 equation *********************************************
//****************************************************************************************************
//****************************************************************************************************
void build_rhs(fftw_complex *temp1,fftw_complex *temp2,fftw_complex *temp3,fftw_complex *temp4,fftw_complex *Vx_N,fftw_complex *Vy_N,fftw_complex *Vz_N,fftw_complex *PI,double *D2, fftw_complex *temp_rhs){
  double * temp_v1, *temp_v2;
  temp_v1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_v2 = fftw_alloc_real(sizeof(double)*Nz);
  for(int i=0;i<Nz;i++){
    temp_v1[i] = 0.0;
    temp_v2[i] = 0.0;
  }
  double *phii;
  phii = fftw_alloc_real(sizeof(double)*Nz*Nz);
  for(int i=0;i<Nz;i++){
    for(int j=0;j<Nz;j++){
      phii[I(i,j,Nz)] = -nu*dt/2.0* D2[I(i,j,Nz)];
    }
  }
  phii[I(Nz-2,0,Nz)] = 1.0;
  phii[I(Nz-1,0,Nz)] = 1.0;
  
  for(int i=1;i<Nz;i++){
    phii[I(Nz-2,i,Nz)] = 1.0;
    phii[I(Nz-1,i,Nz)] = -1.0*phii[I(Nz-1,i-1,Nz)];
  }

  // we put gradz of PI^ into temp4
  Dz(PI,temp4);

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz-2;l++){
	phii[I(l,l,Nz)] = 1 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]);
      }
      
      //building temp1 which is phii^-1 Vx^(N+1/3) + nu*dt/2 laplacian Vx^N - gradx (pi hat)
      // real part
      for(int l=0;l<Nz;l++){
	temp_v1[l] = Vx_N[I3(i,j,l,Ny/2+1)][0] - (-Kx[i]*PI[I3(i,j,l,Ny/2+1)][1]);
      }
      temp_v1[Nz-1]=0.0;
      temp_v1[Nz-2]=0.0;
      papper_cutter(Nz,phii,temp_v2,temp_v1);
      for(int l=0;l<Nz;l++){
	temp1[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
      }

      // imaginary part
      for(int l=0;l<Nz;l++){
	temp_v1[l] = Vx_N[I3(i,j,l,Ny/2+1)][1] - ( Kx[i]*PI[I3(i,j,l,Ny/2+1)][0]);
      }
      temp_v1[Nz-1]=0.0;
      temp_v1[Nz-2]=0.0;
      papper_cutter(Nz,phii,temp_v2,temp_v1);
      for(int l=0;l<Nz;l++){
	temp1[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
      }
      
      //building temp1 which is phii^-1 Vy^(N+1/3) + nu*dt/2 laplacian Vy^N - grady (pi hat)
      // real part
      for(int l=0;l<Nz;l++){
	temp_v1[l] = Vy_N[I3(i,j,l,Ny/2+1)][0] - ( -Ky[j]*PI[I3(i,j,l,Ny/2+1)][1]);
      }
      temp_v1[Nz-1]=0.0;
      temp_v1[Nz-2]=0.0;
      papper_cutter(Nz,phii,temp_v2,temp_v1);
      for(int l=0;l<Nz;l++){
	temp2[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
      }

      // imaginary part
      for(int l=0;l<Nz;l++){
	temp_v1[l] = Vy_N[I3(i,j,l,Ny/2+1)][1] - ( Ky[j]*PI[I3(i,j,l,Ny/2+1)][0]);
      }
      temp_v1[Nz-1]=0.0;
      temp_v1[Nz-2]=0.0;
      papper_cutter(Nz,phii,temp_v2,temp_v1);
      for(int l=0;l<Nz;l++){
	temp2[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
      }
      
      //building temp1 which is phii^-1 Vz^(N+1/3) + nu*dt/2 laplacian Vz^N - gradz (pi hat)
      // real part
      for(int l=0;l<Nz;l++){
	temp_v1[l] = Vz_N[I3(i,j,l,Ny/2+1)][0] - temp4[I3(i,j,l,Ny/2+1)][0];
      }

      temp_rhs[I3(i,j,0,Ny/2+1)][0] = temp_v1[Nz-1];
      temp_rhs[I3(i,j,1,Ny/2+1)][0] = temp_v1[Nz-2];
      
      temp_v1[Nz-1]=0.0;
      temp_v1[Nz-2]=0.0;
      papper_cutter(Nz,phii,temp_v2,temp_v1);
      for(int l=0;l<Nz;l++){
	temp3[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
      }

      // imaginary part
      for(int l=0;l<Nz;l++){
	temp_v1[l] = Vz_N[I3(i,j,l,Ny/2+1)][1] - temp4[I3(i,j,l,Ny/2+1)][1];
      }

      temp_rhs[I3(i,j,0,Ny/2+1)][1] = temp_v1[Nz-1];
      temp_rhs[I3(i,j,1,Ny/2+1)][1] = temp_v1[Nz-2];

      temp_v1[Nz-1]=0.0;
      temp_v1[Nz-2]=0.0;
      papper_cutter(Nz,phii,temp_v2,temp_v1);
      for(int l=0;l<Nz;l++){
	temp3[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
      }

    }
  }

}

//****************************************************************************************************
//****************************************************************************************************
//****************************************************************************************************
//************************* Solving for zero-zero mode, inverting Phi ********************************
//****************************************************************************************************
//****************************************************************************************************
void  zero_zero_mode_solve(fftw_complex *temp1,fftw_complex *temp2, fftw_complex *temp3, fftw_complex *Vx_N, fftw_complex *Vy_N, fftw_complex *Vz_N, double *D2){
  double * temp_v1, *temp_v2;
  temp_v1 = fftw_alloc_real(sizeof(double)*Nz);
  temp_v2 = fftw_alloc_real(sizeof(double)*Nz);
  for(int i=0;i<Nz;i++){
    temp_v1[i] = 0.0;
    temp_v2[i] = 0.0;
  }
  double *phii;
  phii = fftw_alloc_real(sizeof(double)*Nz*Nz);
  for(int i=0;i<Nz;i++){
    for(int j=0;j<Nz;j++){
      phii[I(i,j,Nz)] = -nu*dt/2.0* D2[I(i,j,Nz)];
    }
  }
  phii[I(Nz-2,0,Nz)] = 1.0;
  phii[I(Nz-1,0,Nz)] = 1.0;
  
  for(int i=1;i<Nz;i++){
    phii[I(Nz-2,i,Nz)] = 1.0;
    phii[I(Nz-1,i,Nz)] = -1.0*phii[I(Nz-1,i-1,Nz)];
  }
  //  for(int i=0;i<Nx;i++){
  // for(int j=0;j<Ny/2+1;j++){
  int i=0;
  int j=0;
  for(int l=0;l<Nz-2;l++){
    phii[I(l,l,Nz)] = 1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]);
  }
  
  //building temp1 which is phii^-1 Vx^(N+1/3) + nu*dt/2 laplacian Vx^N - gradx (pi hat)
  // real part
  for(int l=0;l<Nz;l++){
    temp_v1[l] = Vx_N[I3(i,j,l,Ny/2+1)][0];
  }
  temp_v1[Nz-1]=0.0;
  temp_v1[Nz-2]=0.0;
  papper_cutter(Nz,phii,temp_v2,temp_v1);
  for(int l=0;l<Nz;l++){
    temp1[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
  }
  
  // imaginary part
  for(int l=0;l<Nz;l++){
    temp_v1[l] = Vx_N[I3(i,j,l,Ny/2+1)][1];
  }
  temp_v1[Nz-1]=0.0;
  temp_v1[Nz-2]=0.0;
  papper_cutter(Nz,phii,temp_v2,temp_v1);
  for(int l=0;l<Nz;l++){
    temp1[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
  }
  
  //building temp1 which is phii^-1 Vy^(N+1/3) + nu*dt/2 laplacian Vy^N - grady (pi hat)
  // real part
  for(int l=0;l<Nz;l++){
    temp_v1[l] = Vy_N[I3(i,j,l,Ny/2+1)][0];
  }
  temp_v1[Nz-1]=0.0;
  temp_v1[Nz-2]=0.0;
  papper_cutter(Nz,phii,temp_v2,temp_v1);
  for(int l=0;l<Nz;l++){
    temp2[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
  }
  
  // imaginary part
  for(int l=0;l<Nz;l++){
    temp_v1[l] = Vy_N[I3(i,j,l,Ny/2+1)][1];
  }
  temp_v1[Nz-1]=0.0;
  temp_v1[Nz-2]=0.0;
  papper_cutter(Nz,phii,temp_v2,temp_v1);
  for(int l=0;l<Nz;l++){
    temp2[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
  }
  
  //building temp1 which is phii^-1 Vz^(N+1/3) + nu*dt/2 laplacian Vz^N - gradz (pi hat)
  // real part
  for(int l=0;l<Nz;l++){
    temp_v1[l] = Vz_N[I3(i,j,l,Ny/2+1)][0];
  }
  
  temp_v1[Nz-1]=0.0;
  temp_v1[Nz-2]=0.0;
  papper_cutter(Nz,phii,temp_v2,temp_v1);
  for(int l=0;l<Nz;l++){
    temp3[I3(i,j,l,Ny/2+1)][0] = temp_v2[l];
  }
  
  // imaginary part
  for(int l=0;l<Nz;l++){
    temp_v1[l] = Vz_N[I3(i,j,l,Ny/2+1)][1];
  }
  
  temp_v1[Nz-1]=0.0;
  temp_v1[Nz-2]=0.0;
  papper_cutter(Nz,phii,temp_v2,temp_v1);
  for(int l=0;l<Nz;l++){
    temp3[I3(i,j,l,Ny/2+1)][1] = temp_v2[l];
  }

  //  }
  //}

}
//****************************************************************************************************
//****************************************************************************************************
//****************************************************************************************************
//**************************** Computing the inf norm ************************************************
//****************************************************************************************************
//****************************************************************************************************
double norm_infinity(double *d_temp2){
  double max = -10000;
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	if(max < d_temp2[I3(i,j,l,Ny)])
	  max = d_temp2[I3(i,j,l,Ny)];
      }
    }
  }
  return max;
}
//****************************************************************************************************
//****************************************************************************************************
//****************************************************************************************************
//************************************* MAIN FUNCTION*************************************************
//****************************************************************************************************
//****************************************************************************************************
//****************************************************************************************************
int main(){

  ofstream out1;
  out1.open("divergence.dat");
  ofstream out2;
  out2.open("Energy.dat");
  ofstream out3;
  out3.open("velocity_profile.dat");
  double energy;
  ofstream out4;
  out4.open("Vz_end_points.dat");


  cout<<"Initializing constant values..."<<endl;
  //************************ Initializing Kx and Ky **************************************************
  Kx = fftw_alloc_real(sizeof(double)*Nx);
  Ky = fftw_alloc_real(sizeof(double)*Ny);
  
  for ( int i=0;i<Nx/2;i++) Kx[i] = 2.0*M_PI*(double)i/Lx;
  for ( int i=-Nx/2;i<0;i++) Kx[i+Nx] = 2.0*M_PI*(double)i/Lx;
  //Kx[Nx/2]=0.0;

  /*
  for(int i=0;i<Nx;i++)
    cout<<Kx[i]<<"\t";
  cout<<"\n";
  */

  for ( int i=0;i<Ny/2;i++) Ky[i] =  2.0*M_PI*(double)i/Ly;
  for ( int i=-Ny/2;i<0;i++) Ky[i+Ny] = 2.0*M_PI*(double)i/Ly;
  //Ky[Ny/2]=0.0;
  double time=0.0;
  cout<<"Allocating memory..."<<endl;
  double *D2,*d_Vx_N_1,*d_Vy_N_1,*d_Vz_N_1,*d_Wx,*d_Wy,*d_Wz,*d_Vx_N,*d_Vy_N,*d_Vz_N;
  double *d_temp1, *d_temp2;
  double *d_G1,*d_G2,*d_G7,*d_G8;
  double *d_gradz_G1,*d_gradz_G2,*d_gradz_G7,*d_gradz_G8;
  double *d_PI;
  double *d_VWx,*d_VWy,*d_VWz;
  double *dTm,*dTm_1,*dT_temp;
  mat TAU(4,4);
  vec RHS(4);
  vec SOL(4);
  
  fftw_complex *Vx_N_1,*Vy_N_1,*Vz_N_1,*Vx_N,*Vy_N,*Vz_N ;
  fftw_complex *tempVx, *tempVy, *tempVz;
  fftw_complex *VWx_N_1,*VWy_N_1,*VWz_N_1,*VWx_N,*VWy_N,*VWz_N;
  fftw_complex *temp1,*temp2,*temp3,*temp4;
  fftw_complex *PI;
  fftw_complex *phi_inv_grad_G1x, *phi_inv_grad_G2x, *phi_inv_grad_G7x, *phi_inv_grad_G8x;
  fftw_complex *phi_inv_grad_G1y, *phi_inv_grad_G2y, *phi_inv_grad_G7y, *phi_inv_grad_G8y;
  fftw_complex *phi_inv_grad_G1z, *phi_inv_grad_G2z, *phi_inv_grad_G7z, *phi_inv_grad_G8z;
  fftw_complex *gradz_phi_inv_grad_G1z, *gradz_phi_inv_grad_G2z, *gradz_phi_inv_grad_G7z, *gradz_phi_inv_grad_G8z;
  fftw_complex *temp_rhs;
  
  // fftw_complex *Wx,*Wy,*Wz;
  double *c_tau1, *c_tau2, *c_tau7, *c_tau8;
  fftw_complex tau1, tau2, tau7, tau8;
  
  //this will contain gradz(Phi hat), M-1 element for each i and j mode in fourier 
  fftw_complex *Dz_phi_hat_M_1;
  Dz_phi_hat_M_1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1));


  D2 = fftw_alloc_real(sizeof(double)*Nz*Nz);
  d_Vx_N_1 = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_Vy_N_1 = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_Vz_N_1 = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);

  d_Vx_N = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_Vy_N = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_Vz_N = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);

  d_Wx = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_Wy = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_Wz = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);

  d_temp1 = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_temp2 = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);

  dTm = fftw_alloc_real(sizeof(double)*Nz);
  dTm_1 = fftw_alloc_real(sizeof(double)*Nz);
  dT_temp = fftw_alloc_real(sizeof(double)*Nz);

  d_G1 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);
  d_G2 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);
  d_G7 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);
  d_G8 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);

  d_gradz_G1 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);
  d_gradz_G2 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);
  d_gradz_G7 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);
  d_gradz_G8 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*Nz);

  d_PI = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);

  d_VWx = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_VWy = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  d_VWz = fftw_alloc_real(sizeof(double)*Nx*Ny*Nz);
  
  c_tau1 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*4);
  c_tau2 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*4);
  c_tau7 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*4);
  c_tau8 = fftw_alloc_real(sizeof(double)*Nx*(Ny/2+1)*4);
  

  Vx_N_1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  Vy_N_1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  Vz_N_1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  Vx_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  Vy_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  Vz_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  //  Wx = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  //  Wy = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  //  Wz = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  temp1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  temp2 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  temp3 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  temp4 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  VWx_N_1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  VWy_N_1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  VWz_N_1 = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  VWx_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  VWy_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  VWz_N = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  PI = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G1x = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G2x = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G7x = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G8x = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);

  phi_inv_grad_G1y = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G2y = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G7y = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G8y = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);

  phi_inv_grad_G1z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G2z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G7z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  phi_inv_grad_G8z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);

  gradz_phi_inv_grad_G1z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  gradz_phi_inv_grad_G2z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  gradz_phi_inv_grad_G7z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  gradz_phi_inv_grad_G8z = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);

  tempVx = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  tempVy = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);
  tempVz = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*Nz);

  temp_rhs = fftw_alloc_complex(sizeof(fftw_complex)*Nx*(Ny/2+1)*2);

  cout<<"Building laplacian matrix in chebyshev space ..."<<endl;
  //******************** Building D2 matrix, It will be used for building Theta **********************
  build_D2(Nz,D2);
  cout<<"Building G1, G2, G7 and G8 ...."<<endl;
  //******************* Building G1-G2-G7 and G8 for each mode i and j, used in future ***************
  build_G1278(D2,d_G1,d_G2,d_G7,d_G8);

  /*
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	cout<<i<<"\t"<<j<<endl;
	cout<<d_G1[I3(i,j,l,Ny/2+1)]<<endl;}
      cout<<"\n\n";
    }
    cout<<"\n\n";
  }
  */

  
  cout<<"Setting the initial condition ..."<<endl;
  //Initializing the initial condition ...... Vx, Vy, Vz, P
  //Initializing the d_Vx_N_1, d_Vy_N_1, d_Vz_N_1
  double zz;
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	zz= cos(M_PI*(double)l/(double)(Nz-1));
	d_Vx_N_1[I3(i,j,l,Ny)] = (1+zz)*(1-zz);//sin(M_PI*(double)i/(double)(Nx)) ; //sin(2.0*M_PI*(double)j/(double)(Nx));
	d_Vy_N_1[I3(i,j,l,Ny)] = 0.0;//(1-zz)*(1+zz);//sin(M_PI*(double)j/(double)(Ny));//0.0;
	d_Vz_N_1[I3(i,j,l,Ny)] = 0.0;//(1+zz)*(1-zz);//0.0;
	d_PI[I3(i,j,l,Ny)] = 0.0 ;//  + 0.5* (d_Vx_N_1[I3(i,j,l,Ny)]*d_Vx_N_1[I3(i,j,l,Ny)] + d_Vy_N_1[I3(i,j,l,Ny)]*d_Vy_N_1[I3(i,j,l,Ny)] + d_Vz_N_1[I3(i,j,l,Ny)]*d_Vz_N_1[I3(i,j,l,Ny)]);
	
      }
    }
  }
  
  /*
  for(int j=0;j<Ny;j++){
    for(int l=0;l<Nz;l++){
      d_Vx_N_1[I3(Nx-1,j,l,Ny)] = 0.0;
      //d_Vx_N_1[I3(0 ,j,l,Ny)] = 0.0;
    }
  }
  */

  
  cout<<"Calculating the first step using second order Ronge-Kutta ..."<<endl;
  CFF_transformer_forward(d_Vx_N_1,Vx_N_1);
  CFF_transformer_forward(d_Vy_N_1,Vy_N_1);
  CFF_transformer_forward(d_Vz_N_1,Vz_N_1);
  CFF_transformer_forward(d_PI,PI);

  omega_calculator(d_Wx,d_Wy,d_Wz,d_temp1,d_temp2,Vx_N_1,Vy_N_1,Vz_N_1,temp1,temp2);
  cross_product(d_Vx_N_1,d_Vy_N_1,d_Vz_N_1, d_Wx, d_Wy, d_Wz, d_VWx, d_VWy, d_VWz, VWx_N_1, VWy_N_1, VWz_N_1);
    


  //Doing Runge-Kutta method 
  // We will firs compute y_n + 1/2 dt f(tn,yn)
  laplacian(Vx_N_1,temp1);
  laplacian(Vy_N_1,temp2);
  laplacian(Vz_N_1,temp3);

  Dz(PI,temp4);
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	Vx_N[I3(i,j,l,Ny/2+1)][0] = Vx_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0 *( VWx_N_1[I3(i,j,l,Ny/2+1)][0] -(-Kx[i]*PI[I3(i,j,l,Ny/2+1)][1]) + nu*temp1[I3(i,j,l,Ny/2+1)][0]);
	Vx_N[I3(i,j,l,Ny/2+1)][1] = Vx_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0 *( VWx_N_1[I3(i,j,l,Ny/2+1)][1] -( Kx[i]*PI[I3(i,j,l,Ny/2+1)][0]) + nu*temp1[I3(i,j,l,Ny/2+1)][1]);

	Vy_N[I3(i,j,l,Ny/2+1)][0] = Vy_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0 *( VWy_N_1[I3(i,j,l,Ny/2+1)][0] -(-Ky[j]*PI[I3(i,j,l,Ny/2+1)][1]) + nu*temp2[I3(i,j,l,Ny/2+1)][0]);
	Vy_N[I3(i,j,l,Ny/2+1)][1] = Vy_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0 *( VWy_N_1[I3(i,j,l,Ny/2+1)][1] -(+Ky[j]*PI[I3(i,j,l,Ny/2+1)][0]) + nu*temp2[I3(i,j,l,Ny/2+1)][1]);
	
	Vz_N[I3(i,j,l,Ny/2+1)][0] = Vz_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0 *( VWz_N_1[I3(i,j,l,Ny/2+1)][0] -( temp4[I3(i,j,l,Ny/2+1)][0]) + nu*temp3[I3(i,j,l,Ny/2+1)][0]);
	Vz_N[I3(i,j,l,Ny/2+1)][1] = Vz_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0 *( VWz_N_1[I3(i,j,l,Ny/2+1)][1] -( temp4[I3(i,j,l,Ny/2+1)][1]) + nu*temp3[I3(i,j,l,Ny/2+1)][1]);
      }
    }
  }

  CFF_transformer_backward(d_Vx_N, Vx_N);
  CFF_transformer_backward(d_Vy_N, Vy_N);
  CFF_transformer_backward(d_Vz_N, Vz_N);

  //changing PI since PI = p + 1/2 V*V
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	d_PI[I3(i,j,l,Ny)] = 0.0 ; //d_PI[I3(i,j,l,Ny)] - 0.5* (d_Vx_N_1[I3(i,j,l,Ny)]*d_Vx_N_1[I3(i,j,l,Ny)] + d_Vy_N_1[I3(i,j,l,Ny)]*d_Vy_N_1[I3(i,j,l,Ny)] + d_Vz_N_1[I3(i,j,l,Ny)]*d_Vz_N_1[I3(i,j,l,Ny)]) ; 
	d_PI[I3(i,j,l,Ny)] = 0.0 ;// d_PI[I3(i,j,l,Ny)] + 0.5* (d_Vx_N[I3(i,j,l,Ny)]*d_Vx_N[I3(i,j,l,Ny)] + d_Vy_N[I3(i,j,l,Ny)]*d_Vy_N[I3(i,j,l,Ny)] + d_Vz_N[I3(i,j,l,Ny)]*d_Vz_N[I3(i,j,l,Ny)]);
      }
    }
  }
  CFF_transformer_forward(d_PI,PI);
  
  omega_calculator(d_Wx,d_Wy,d_Wz,d_temp1,d_temp2,Vx_N,Vy_N,Vz_N,temp1,temp2);
  cross_product(d_Vx_N, d_Vy_N, d_Vz_N, d_Wx, d_Wy, d_Wz, d_VWx, d_VWy, d_VWz, VWx_N, VWy_N, VWz_N);
  
  laplacian(Vx_N_1,temp1);
  laplacian(Vy_N_1,temp2);
  laplacian(Vz_N_1,temp3);
  Dz(PI,temp4);

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	Vx_N[I3(i,j,l,Ny/2+1)][0] = Vx_N[I3(i,j,l,Ny/2+1)][0] + dt *( VWx_N[I3(i,j,l,Ny/2+1)][0] -(-Kx[i]*PI[I3(i,j,l,Ny/2+1)][1]) + nu*temp1[I3(i,j,l,Ny/2+1)][0]);
	Vx_N[I3(i,j,l,Ny/2+1)][1] = Vx_N[I3(i,j,l,Ny/2+1)][1] + dt *( VWx_N[I3(i,j,l,Ny/2+1)][1] -( Kx[i]*PI[I3(i,j,l,Ny/2+1)][0]) + nu*temp1[I3(i,j,l,Ny/2+1)][1]);

	Vy_N[I3(i,j,l,Ny/2+1)][0] = Vy_N[I3(i,j,l,Ny/2+1)][0] + dt *( VWy_N[I3(i,j,l,Ny/2+1)][0] -(-Ky[j]*PI[I3(i,j,l,Ny/2+1)][1]) + nu*temp2[I3(i,j,l,Ny/2+1)][0]);
	Vy_N[I3(i,j,l,Ny/2+1)][1] = Vy_N[I3(i,j,l,Ny/2+1)][1] + dt *( VWy_N[I3(i,j,l,Ny/2+1)][1] -( Ky[j]*PI[I3(i,j,l,Ny/2+1)][0]) + nu*temp2[I3(i,j,l,Ny/2+1)][1]);
	
	Vz_N[I3(i,j,l,Ny/2+1)][0] = Vz_N[I3(i,j,l,Ny/2+1)][0] + dt *( VWz_N[I3(i,j,l,Ny/2+1)][0] -( temp4[I3(i,j,l,Ny/2+1)][0]) + nu*temp3[I3(i,j,l,Ny/2+1)][0]);
	Vz_N[I3(i,j,l,Ny/2+1)][1] = Vz_N[I3(i,j,l,Ny/2+1)][1] + dt *( VWz_N[I3(i,j,l,Ny/2+1)][1] -( temp4[I3(i,j,l,Ny/2+1)][1]) + nu*temp3[I3(i,j,l,Ny/2+1)][1]);
      }
    }
  }

  //So far we have  d_V^N-1, V^N-1, (VxW)^N-1 , V^N
  
  CFF_transformer_backward(d_Vx_N, Vx_N);
  CFF_transformer_backward(d_Vy_N, Vy_N);
  CFF_transformer_backward(d_Vz_N, Vz_N);
  

  omega_calculator(d_Wx,d_Wy,d_Wz,d_temp1,d_temp2,Vx_N,Vy_N,Vz_N,temp1,temp2);
  cross_product(d_Vx_N, d_Vy_N, d_Vz_N, d_Wx, d_Wy, d_Wz, d_VWx, d_VWy, d_VWz, VWx_N, VWy_N, VWz_N);



  // now we have d_V^N, V^N and (VxW)^N  and d_V^N-1, V^N-1 and (VxW)^N-1
  // but in our calculation we will only need (VxW)^N and (VxW)^N-1 and V^N
  // Now we copy back data of V^N to V^N-1 for both physical and CFF space

  
  /*
  //computing to see if the field is divergence free   
  Dz(Vz_N,temp1);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	temp2[I3(i,j,l,Ny/2+1)][0] = - Kx[i]*Vx_N[I3(i,j,l,Ny/2+1)][1] - Ky[j]*Vy_N[I3(i,j,l,Ny/2+1)][1] + temp1[I3(i,j,l,Ny/2+1)][0];
	temp2[I3(i,j,l,Ny/2+1)][1] =   Kx[i]*Vx_N[I3(i,j,l,Ny/2+1)][0] + Ky[j]*Vy_N[I3(i,j,l,Ny/2+1)][0] + temp1[I3(i,j,l,Ny/2+1)][1];
      }
    }
  }
  CFF_transformer_backward(d_temp2, temp2);
  //  for(int i=0;i<Nx;i++)
  //  for(int j=0;j<Ny;j++)
  for(int l=0;l<Nz;l++){
    cout<<d_temp2[I3(2,2,l,Ny)]<<"\n";
  }
  // up to here for computing and giving the divergence    
  */




  /*
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	d_Vx_N_1[I3(i,j,l,Ny)] = d_Vx_N[I3(i,j,l,Ny)];
	d_Vy_N_1[I3(i,j,l,Ny)] = d_Vy_N[I3(i,j,l,Ny)];
	d_Vz_N_1[I3(i,j,l,Ny)] = d_Vz_N[I3(i,j,l,Ny)];
      }
    }
  }

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	Vx_N_1[I3(i,j,l,Ny/2+1)][0] = Vx_N[I3(i,j,l,Ny/2+1)][0];
	Vx_N_1[I3(i,j,l,Ny/2+1)][1] = Vx_N[I3(i,j,l,Ny/2+1)][1];

	Vy_N_1[I3(i,j,l,Ny/2+1)][0] = Vy_N[I3(i,j,l,Ny/2+1)][0];
	Vy_N_1[I3(i,j,l,Ny/2+1)][1] = Vy_N[I3(i,j,l,Ny/2+1)][1];

	Vz_N_1[I3(i,j,l,Ny/2+1)][0] = Vz_N[I3(i,j,l,Ny/2+1)][0];
	Vz_N_1[I3(i,j,l,Ny/2+1)][1] = Vz_N[I3(i,j,l,Ny/2+1)][1];
      }
    }
  }
  */
  //Now we have all the data that we want to start our for loop on time 0

  // For moving forward we need to find c_tau1, c_tau2, c_tau7, c_tau8
  // these coefficients are computed once and used in the main loop
  // d/dz of G1, G2, G7 and G8 are stored in temp1, temp2, temp3 and temp4 respectively
  cout<<"Computing coefficient of Tau1, Tau2, Tau7 and Tau8 ..."<<endl;

  Dz_real(d_G1, d_gradz_G1);
  Dz_real(d_G2, d_gradz_G2);
  Dz_real(d_G7, d_gradz_G7);
  Dz_real(d_G8, d_gradz_G8);
  
  /*
  // printing out the Gibbs functions
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	cout<<d_gradz_G8[I3(i,j,l,Ny/2+1)]<<endl;
      }
      cout<<"\n\n";
    }
  }
  */


  // This all should be removed 
  Ai(Nz,dTm_1,Nz-2);
  Ai(Nz,dTm,Nz-1);
  chebyshev_derivative(Nz,dTm_1,dT_temp);
  for(int l=0;l<Nz;l++){
    dTm_1[l] = dT_temp[l];
  }
  chebyshev_derivative(Nz,dTm,dT_temp);
  for(int l=0;l<Nz;l++){
    dTm[l] = dT_temp[l];
  }
  // up to here 

  compute_phi_inv_grad_G(phi_inv_grad_G1x, phi_inv_grad_G2x, phi_inv_grad_G7x, phi_inv_grad_G8x,phi_inv_grad_G1y, phi_inv_grad_G2y, phi_inv_grad_G7y, phi_inv_grad_G8y,phi_inv_grad_G1z, phi_inv_grad_G2z, phi_inv_grad_G7z, phi_inv_grad_G8z, D2, d_G1, d_G2, d_G7, d_G8, d_gradz_G1, d_gradz_G2, d_gradz_G7, d_gradz_G8);

  /*
  for(int l=0;l<Nz;l++)
    cout<<d_G1[I3(2,1,l,Ny/2+1)]<<"\t"<<phi_inv_grad_G1z[I3(2,1,l,Ny/2+1)][0]<<endl;
  cout<<"\n\n\n";
  */
  // for debugging mode
  // cout<<"last element of Tm is:   "<<dTm[Nz-2]<<endl; //cout<<"last element of Tm-1 is: "<<dTm_1[Nz-2]<<endl;

  //taking d/dz of the phi_inv_grad_Giz
  Dz(phi_inv_grad_G1z,gradz_phi_inv_grad_G1z);
  Dz(phi_inv_grad_G2z,gradz_phi_inv_grad_G2z);
  Dz(phi_inv_grad_G7z,gradz_phi_inv_grad_G7z);
  Dz(phi_inv_grad_G8z,gradz_phi_inv_grad_G8z);


  /*
  for(int l=0;l<Nz;l++)
    cout<<phi_inv_grad_G1z[I3(2,1,l,Ny/2+1)][0]<<"\t"<<gradz_phi_inv_grad_G1z[I3(2,1,l,Ny/2+1)][0]<<endl;
  cout<<"\n";
  */
  
  // tau from 0 to 4 is basically solving for V^N+1 highest modes, mode M and M-1, and then divergence free of V^N+1 for highest modes, mode M and M-1


  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      if(i!=0 | j!=0){
	//c_tau1[I3(i,j,0,Ny/2+1)] = 0.0;
	//c_tau1[I3(i,j,1,Ny/2+1)] = d_gradz_G1[I3(i,j,Nz-2,Ny/2+1)] -1.0;
	c_tau1[I3(i,j,0,Ny/2+1)] = phi_inv_grad_G1z[I3(i,j,Nz-1,Ny/2+1)][0]  - (d_gradz_G1[I3(i,j,Nz-1,Ny/2+1)])/(1.0+ nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j])); 
	c_tau1[I3(i,j,1,Ny/2+1)] = phi_inv_grad_G1z[I3(i,j,Nz-2,Ny/2+1)][0]  - (d_gradz_G1[I3(i,j,Nz-2,Ny/2+1)]-1.0)/(1.0+ nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	c_tau1[I3(i,j,2,Ny/2+1)] = -Kx[i] * phi_inv_grad_G1x[I3(i,j,Nz-1,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G1y[I3(i,j,Nz-1,Ny/2+1)][1] + gradz_phi_inv_grad_G1z[I3(i,j,Nz-1,Ny/2+1)][0];
	c_tau1[I3(i,j,3,Ny/2+1)] = -Kx[i] * phi_inv_grad_G1x[I3(i,j,Nz-2,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G1y[I3(i,j,Nz-2,Ny/2+1)][1] + gradz_phi_inv_grad_G1z[I3(i,j,Nz-2,Ny/2+1)][0];
	
	//c_tau2[I3(i,j,0,Ny/2+1)] = 1.0;
	//c_tau2[I3(i,j,1,Ny/2+1)] = d_gradz_G2[I3(i,j,Nz-2,Ny/2+1)];
	c_tau2[I3(i,j,0,Ny/2+1)] = phi_inv_grad_G2z[I3(i,j,Nz-1,Ny/2+1)][0] - (d_gradz_G2[I3(i,j,Nz-1,Ny/2+1)]-1.0)/(1.0+ nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	c_tau2[I3(i,j,1,Ny/2+1)] = phi_inv_grad_G2z[I3(i,j,Nz-2,Ny/2+1)][0] - (d_gradz_G2[I3(i,j,Nz-2,Ny/2+1)])/(1+ nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	c_tau2[I3(i,j,2,Ny/2+1)] = -Kx[i] * phi_inv_grad_G2x[I3(i,j,Nz-1,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G2y[I3(i,j,Nz-1,Ny/2+1)][1] + gradz_phi_inv_grad_G2z[I3(i,j,Nz-1,Ny/2+1)][0];
	c_tau2[I3(i,j,3,Ny/2+1)] = -Kx[i] * phi_inv_grad_G2x[I3(i,j,Nz-2,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G2y[I3(i,j,Nz-2,Ny/2+1)][1] + gradz_phi_inv_grad_G2z[I3(i,j,Nz-2,Ny/2+1)][0];
	
	//c_tau7[I3(i,j,0,Ny/2+1)] = 0.0;
	//c_tau7[I3(i,j,1,Ny/2+1)] = d_gradz_G7[I3(i,j,Nz-2,Ny/2+1)];
	c_tau7[I3(i,j,0,Ny/2+1)] = phi_inv_grad_G7z[I3(i,j,Nz-1,Ny/2+1)][0] - (d_gradz_G7[I3(i,j,Nz-1,Ny/2+1)])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));;
	c_tau7[I3(i,j,1,Ny/2+1)] = phi_inv_grad_G7z[I3(i,j,Nz-2,Ny/2+1)][0] - (d_gradz_G7[I3(i,j,Nz-2,Ny/2+1)])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));;
	c_tau7[I3(i,j,2,Ny/2+1)] = -Kx[i] * phi_inv_grad_G7x[I3(i,j,Nz-1,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G7y[I3(i,j,Nz-1,Ny/2+1)][1] + gradz_phi_inv_grad_G7z[I3(i,j,Nz-1,Ny/2+1)][0];
	c_tau7[I3(i,j,3,Ny/2+1)] = -Kx[i] * phi_inv_grad_G7x[I3(i,j,Nz-2,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G7y[I3(i,j,Nz-2,Ny/2+1)][1] + gradz_phi_inv_grad_G7z[I3(i,j,Nz-2,Ny/2+1)][0];
	
	//c_tau8[I3(i,j,0,Ny/2+1)] = 0.0;
	//c_tau8[I3(i,j,1,Ny/2+1)] = d_gradz_G8[I3(i,j,Nz-2,Ny/2+1)];
	c_tau8[I3(i,j,0,Ny/2+1)] = phi_inv_grad_G8z[I3(i,j,Nz-1,Ny/2+1)][0] - (d_gradz_G8[I3(i,j,Nz-1,Ny/2+1)])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));;
	c_tau8[I3(i,j,1,Ny/2+1)] = phi_inv_grad_G8z[I3(i,j,Nz-2,Ny/2+1)][0] - (d_gradz_G8[I3(i,j,Nz-2,Ny/2+1)])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));;
	c_tau8[I3(i,j,2,Ny/2+1)] = -Kx[i] * phi_inv_grad_G8x[I3(i,j,Nz-1,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G8y[I3(i,j,Nz-1,Ny/2+1)][1] + gradz_phi_inv_grad_G8z[I3(i,j,Nz-1,Ny/2+1)][0];
	c_tau8[I3(i,j,3,Ny/2+1)] = -Kx[i] * phi_inv_grad_G8x[I3(i,j,Nz-2,Ny/2+1)][1] - Ky[j]*phi_inv_grad_G8y[I3(i,j,Nz-2,Ny/2+1)][1] + gradz_phi_inv_grad_G8z[I3(i,j,Nz-2,Ny/2+1)][0];
      }
    }
  }
  

  //Now we are set to compute the do loop for the whole time 
  //first we will do the first time step
  
  //******************************************** Making the first step divergence free



  laplacian(Vx_N_1,Vx_N);
  laplacian(Vy_N_1,Vy_N);
  laplacian(Vz_N_1,Vz_N);
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	Vx_N[I3(i,j,l,Ny/2+1)][0] = Vx_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0*(3*VWx_N[I3(i,j,l,Ny/2+1)][0] - VWx_N_1[I3(i,j,l,Ny/2+1)][0]) + nu*dt/2.0*Vx_N[I3(i,j,l,Ny/2+1)][0];
	Vx_N[I3(i,j,l,Ny/2+1)][1] = Vx_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0*(3*VWx_N[I3(i,j,l,Ny/2+1)][1] - VWx_N_1[I3(i,j,l,Ny/2+1)][1]) + nu*dt/2.0*Vx_N[I3(i,j,l,Ny/2+1)][1];

	Vy_N[I3(i,j,l,Ny/2+1)][0] = Vy_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0*(3*VWy_N[I3(i,j,l,Ny/2+1)][0] - VWy_N_1[I3(i,j,l,Ny/2+1)][0]) + nu*dt/2.0*Vy_N[I3(i,j,l,Ny/2+1)][0];
	Vy_N[I3(i,j,l,Ny/2+1)][1] = Vy_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0*(3*VWy_N[I3(i,j,l,Ny/2+1)][1] - VWy_N_1[I3(i,j,l,Ny/2+1)][1]) + nu*dt/2.0*Vy_N[I3(i,j,l,Ny/2+1)][1];

	Vz_N[I3(i,j,l,Ny/2+1)][0] = Vz_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0*(3*VWz_N[I3(i,j,l,Ny/2+1)][0] - VWz_N_1[I3(i,j,l,Ny/2+1)][0]) + nu*dt/2.0*Vz_N[I3(i,j,l,Ny/2+1)][0];
	Vz_N[I3(i,j,l,Ny/2+1)][1] = Vz_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0*(3*VWz_N[I3(i,j,l,Ny/2+1)][1] - VWz_N_1[I3(i,j,l,Ny/2+1)][1]) + nu*dt/2.0*Vz_N[I3(i,j,l,Ny/2+1)][1];
      }
    }
  }
  // Now our V^N contains V^(N+1/3) + nu*dt/2* laplacian (V^N)
  
  // Now its time to build Pi_hat
  build_pi_hat(Vx_N,Vy_N,Vz_N,D2,PI,temp1);
 

  build_rhs(temp1,temp2,temp3,temp4,Vx_N,Vy_N,Vz_N,PI,D2,temp_rhs);
  // up to now, Pi is Pi^ 
  // temp1 =  phii^-1 ( Vx^(N+1/3) + nu*dt/2 laplacian Vx^N - gradx (pi hat))
  // temp2 =  phii^-1 ( Vy^(N+1/3) + nu*dt/2 laplacian Vy^N - grady (pi hat))
  // temp3 =  phii^-1 ( Vz^(N+1/3) + nu*dt/2 laplacian Vz^N - gradz (pi hat))
  // temp4 =  Dz(PI^)
  // we will need Dz(PI^), M-1 element which is Nz-2
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      Dz_phi_hat_M_1[I(i,j,Ny/2+1)][0] = temp4[I3(i,j,Nz-2,Ny/2+1)][0];
      Dz_phi_hat_M_1[I(i,j,Ny/2+1)][1] = temp4[I3(i,j,Nz-2,Ny/2+1)][1];
    }
  }

  Dz(temp3,temp4);
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      if( i!=0 | j !=0){
	// copy tau matrix that we built into TAU matrix 
	for(int l=0;l<4;l++)
	  TAU(l,0) = c_tau1[I3(i,j,l,Ny/2+1)];
	for(int l=0;l<4;l++)
	  TAU(l,1) = c_tau2[I3(i,j,l,Ny/2+1)];
	for(int l=0;l<4;l++)
	  TAU(l,2) = c_tau7[I3(i,j,l,Ny/2+1)];
	for(int l=0;l<4;l++)
	  TAU(l,3) = c_tau8[I3(i,j,l,Ny/2+1)]; 
	//RHS[0] = - (Vz_N[I3(i,j,Nz-1,Ny/2+1)][0]);
	//RHS[1] = Vz_N[I3(i,j,Nz-2,Ny/2+1)][0] -Dz_phi_hat_M_1[I(i,j,Ny/2+1)][0] ;
	RHS[0] = temp3[I3(i,j,Nz-1,Ny/2+1)][0] - (temp_rhs[I3(i,j,0,Ny/2+1)][0])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	RHS[1] = temp3[I3(i,j,Nz-2,Ny/2+1)][0] - (temp_rhs[I3(i,j,1,Ny/2+1)][0])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	RHS[2] = -Kx[i]*temp1[I3(i,j,Nz-1,Ny/2+1)][1] - Ky[j]*temp2[I3(i,j,Nz-1,Ny/2+1)][1] + temp4[I3(i,j,Nz-1,Ny/2+1)][0];
	RHS[3] = -Kx[i]*temp1[I3(i,j,Nz-2,Ny/2+1)][1] - Ky[j]*temp2[I3(i,j,Nz-2,Ny/2+1)][1] + temp4[I3(i,j,Nz-2,Ny/2+1)][0]; 
	//cout<<"i :"<<i<<"\t j :"<<j<<"\n";

	//cout<<c_tau1[I3(i,j,0,Ny/2+1)]<<"\t"<<c_tau1[I3(i,j,1,Ny/2+1)]<<"\t"<<c_tau1[I3(i,j,2,Ny/2+1)]<<"\t"<<c_tau1[I3(i,j,3,Ny/2+1)]<<endl;
	//cout<<"tau is :\n"<<TAU;
	solve(SOL,TAU,RHS);
	//cout<<"RHS is:\n"<<RHS;
	//cout<<"SOL is :\n"<<SOL;
	tau1[0] = SOL(0);
	tau2[0] = SOL(1);
	tau7[0] = SOL(2);
	tau8[0] = SOL(3);
	
	//RHS[0] = - (Vz_N[I3(i,j,Nz-1,Ny/2+1)][1]);
	//RHS[1] = Vz_N[I3(i,j,Nz-2,Ny/2+1)][1] - Dz_phi_hat_M_1[I(i,j,Ny/2+1)][0]; // gradz of PI^ in here is missing 
	RHS[0] = temp3[I3(i,j,Nz-1,Ny/2+1)][1] - (temp_rhs[I3(i,j,0,Ny/2+1)][1])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	RHS[1] = temp3[I3(i,j,Nz-2,Ny/2+1)][1] - (temp_rhs[I3(i,j,1,Ny/2+1)][1])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j])); 
	RHS[2] = Kx[i]*temp1[I3(i,j,Nz-1,Ny/2+1)][0] + Ky[j]*temp2[I3(i,j,Nz-1,Ny/2+1)][0] + temp4[I3(i,j,Nz-1,Ny/2+1)][1];
	RHS[3] = Kx[i]*temp1[I3(i,j,Nz-2,Ny/2+1)][0] + Ky[j]*temp2[I3(i,j,Nz-2,Ny/2+1)][0] + temp4[I3(i,j,Nz-2,Ny/2+1)][1]; 
	//cout<<"i :"<<i<<"\t j :"<<j<<"\n";
	//cout<<TAU;
	solve(SOL,TAU,RHS);
	// cout<<SOL;
	//cout<<"RHS is:\n"<<RHS;
	//cout<<"SOL is :\n"<<SOL;
	tau1[1] = SOL(0);
	tau2[1] = SOL(1);
	tau7[1] = SOL(2);
	tau8[1] = SOL(3);

	for(int l=0;l<Nz;l++){
	  tempVx[I3(i,j,l,Ny/2+1)][0] = temp1[I3(i,j,l,Ny/2+1)][0]  + tau1[1] * phi_inv_grad_G1x[I3(i,j,l,Ny/2+1)][1]  + tau2[1] * phi_inv_grad_G2x[I3(i,j,l,Ny/2+1)][1]  + tau7[1] * phi_inv_grad_G7x[I3(i,j,l,Ny/2+1)][1]  + tau8[1] * phi_inv_grad_G8x[I3(i,j,l,Ny/2+1)][1];

	  tempVx[I3(i,j,l,Ny/2+1)][1] = temp1[I3(i,j,l,Ny/2+1)][1] - tau1[0] * phi_inv_grad_G1x[I3(i,j,l,Ny/2+1)][1]  - tau2[0] * phi_inv_grad_G2x[I3(i,j,l,Ny/2+1)][1]  - tau7[0] * phi_inv_grad_G7x[I3(i,j,l,Ny/2+1)][1]  - tau8[0] * phi_inv_grad_G8x[I3(i,j,l,Ny/2+1)][1] ;
	

	  tempVy[I3(i,j,l,Ny/2+1)][0] = temp2[I3(i,j,l,Ny/2+1)][0]  + tau1[1] * phi_inv_grad_G1y[I3(i,j,l,Ny/2+1)][1]  + tau2[1] * phi_inv_grad_G2y[I3(i,j,l,Ny/2+1)][1]  + tau7[1] * phi_inv_grad_G7y[I3(i,j,l,Ny/2+1)][1]  + tau8[1] * phi_inv_grad_G8y[I3(i,j,l,Ny/2+1)][1];

	  tempVy[I3(i,j,l,Ny/2+1)][1] = temp2[I3(i,j,l,Ny/2+1)][1] - tau1[0] * phi_inv_grad_G1y[I3(i,j,l,Ny/2+1)][1]  - tau2[0] * phi_inv_grad_G2y[I3(i,j,l,Ny/2+1)][1]  - tau7[0] * phi_inv_grad_G7y[I3(i,j,l,Ny/2+1)][1]  - tau8[0] * phi_inv_grad_G8y[I3(i,j,l,Ny/2+1)][1] ;

	  tempVz[I3(i,j,l,Ny/2+1)][0] = temp3[I3(i,j,l,Ny/2+1)][0] - tau1[0] * phi_inv_grad_G1z[I3(i,j,l,Ny/2+1)][0]  - tau2[0] * phi_inv_grad_G2z[I3(i,j,l,Ny/2+1)][0]  - tau7[0] * phi_inv_grad_G7z[I3(i,j,l,Ny/2+1)][0]  - tau8[0] * phi_inv_grad_G8z[I3(i,j,l,Ny/2+1)][0] ;

	  tempVz[I3(i,j,l,Ny/2+1)][1] = temp3[I3(i,j,l,Ny/2+1)][1]  - tau1[1] * phi_inv_grad_G1z[I3(i,j,l,Ny/2+1)][0]  - tau2[1] * phi_inv_grad_G2z[I3(i,j,l,Ny/2+1)][0]  - tau7[1] * phi_inv_grad_G7z[I3(i,j,l,Ny/2+1)][0]  - tau8[1] * phi_inv_grad_G8z[I3(i,j,l,Ny/2+1)][0];


	}

      }  
    }
  }
  //solving the zero zero mode 
  zero_zero_mode_solve(temp1, temp2, temp3, Vx_N, Vy_N, Vz_N, D2);
  for(int l=0;l<Nz;l++){
    tempVz[I3(0,0,l,Ny/2+1)][0] = 0.0;//temp3[I3(0,0,l,Ny/2+1)][0]; //0.0;
    tempVz[I3(0,0,l,Ny/2+1)][1] = 0.0;//;temp3[I3(0,0,l,Ny/2+1)][1];//0.0;
    
    tempVx[I3(0,0,l,Ny/2+1)][0] = temp1[I3(0,0,l,Ny/2+1)][0]; // Vx_N[I3(0,0,l,Ny/2+1)][0];
    tempVx[I3(0,0,l,Ny/2+1)][1] = temp1[I3(0,0,l,Ny/2+1)][1]; //Vx_N[I3(0,0,l,Ny/2+1)][1];
  
    tempVy[I3(0,0,l,Ny/2+1)][0] = temp2[I3(0,0,l,Ny/2+1)][0]; //Vy_N[I3(0,0,l,Ny/2+1)][0];
    tempVy[I3(0,0,l,Ny/2+1)][1] = temp2[I3(0,0,l,Ny/2+1)][1]; //Vy_N[I3(0,0,l,Ny/2+1)][1];
  }
  
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	Vx_N_1[I3(i,j,l,Ny/2+1)][0] = tempVx[I3(i,j,l,Ny/2+1)][0];
	Vx_N_1[I3(i,j,l,Ny/2+1)][1] = tempVx[I3(i,j,l,Ny/2+1)][1];

	Vy_N_1[I3(i,j,l,Ny/2+1)][0] = tempVy[I3(i,j,l,Ny/2+1)][0];
	Vy_N_1[I3(i,j,l,Ny/2+1)][1] = tempVy[I3(i,j,l,Ny/2+1)][1];

	Vz_N_1[I3(i,j,l,Ny/2+1)][0] = tempVz[I3(i,j,l,Ny/2+1)][0];
	Vz_N_1[I3(i,j,l,Ny/2+1)][1] = tempVz[I3(i,j,l,Ny/2+1)][1];
      }
    }
  }
  

  CFF_transformer_backward(d_Vx_N_1,Vx_N_1);
  CFF_transformer_backward(d_Vy_N_1,Vy_N_1);
  CFF_transformer_backward(d_Vz_N_1,Vz_N_1);

  omega_calculator(d_Wx,d_Wy,d_Wz,d_temp1,d_temp2,Vx_N_1,Vy_N_1,Vz_N_1,temp1,temp2);
  cross_product(d_Vx_N_1, d_Vy_N_1, d_Vz_N_1, d_Wx, d_Wy, d_Wz, d_VWx, d_VWy, d_VWz, temp1, temp2, temp3);

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	/*
	VWx_N_1[I3(i,j,l,Ny/2+1)][0] = VWx_N[I3(i,j,l,Ny/2+1)][0];
	VWx_N_1[I3(i,j,l,Ny/2+1)][1] = VWx_N[I3(i,j,l,Ny/2+1)][1];

	VWy_N_1[I3(i,j,l,Ny/2+1)][0] = VWy_N[I3(i,j,l,Ny/2+1)][0];
	VWy_N_1[I3(i,j,l,Ny/2+1)][1] = VWy_N[I3(i,j,l,Ny/2+1)][1];

	VWz_N_1[I3(i,j,l,Ny/2+1)][0] = VWz_N[I3(i,j,l,Ny/2+1)][0];
	VWz_N_1[I3(i,j,l,Ny/2+1)][1] = VWz_N[I3(i,j,l,Ny/2+1)][1];
	*/
	
	VWx_N[I3(i,j,l,Ny/2+1)][0] = temp1[I3(i,j,l,Ny/2+1)][0];
	VWx_N[I3(i,j,l,Ny/2+1)][1] = temp1[I3(i,j,l,Ny/2+1)][1];

	VWy_N[I3(i,j,l,Ny/2+1)][0] = temp2[I3(i,j,l,Ny/2+1)][0];
	VWy_N[I3(i,j,l,Ny/2+1)][1] = temp2[I3(i,j,l,Ny/2+1)][1];

	VWz_N[I3(i,j,l,Ny/2+1)][0] = temp3[I3(i,j,l,Ny/2+1)][0];
	VWz_N[I3(i,j,l,Ny/2+1)][1] = temp3[I3(i,j,l,Ny/2+1)][1];
	
      }
    }
  }





  //***************************************************************************************

  // first we will compute the term Vz^(N+1/3) + nu*dt/2*laplacian V
  // since this is gonna appear everywhere  

  while(time<T){
  
  laplacian(Vx_N_1,Vx_N);
  laplacian(Vy_N_1,Vy_N);
  laplacian(Vz_N_1,Vz_N);
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	Vx_N[I3(i,j,l,Ny/2+1)][0] = Vx_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0*(3*VWx_N[I3(i,j,l,Ny/2+1)][0] - VWx_N_1[I3(i,j,l,Ny/2+1)][0]) + nu*dt/2.0*Vx_N[I3(i,j,l,Ny/2+1)][0];
	Vx_N[I3(i,j,l,Ny/2+1)][1] = Vx_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0*(3*VWx_N[I3(i,j,l,Ny/2+1)][1] - VWx_N_1[I3(i,j,l,Ny/2+1)][1]) + nu*dt/2.0*Vx_N[I3(i,j,l,Ny/2+1)][1];

	Vy_N[I3(i,j,l,Ny/2+1)][0] = Vy_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0*(3*VWy_N[I3(i,j,l,Ny/2+1)][0] - VWy_N_1[I3(i,j,l,Ny/2+1)][0]) + nu*dt/2.0*Vy_N[I3(i,j,l,Ny/2+1)][0];
	Vy_N[I3(i,j,l,Ny/2+1)][1] = Vy_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0*(3*VWy_N[I3(i,j,l,Ny/2+1)][1] - VWy_N_1[I3(i,j,l,Ny/2+1)][1]) + nu*dt/2.0*Vy_N[I3(i,j,l,Ny/2+1)][1];

	Vz_N[I3(i,j,l,Ny/2+1)][0] = Vz_N_1[I3(i,j,l,Ny/2+1)][0] + dt/2.0*(3*VWz_N[I3(i,j,l,Ny/2+1)][0] - VWz_N_1[I3(i,j,l,Ny/2+1)][0]) + nu*dt/2.0*Vz_N[I3(i,j,l,Ny/2+1)][0];
	Vz_N[I3(i,j,l,Ny/2+1)][1] = Vz_N_1[I3(i,j,l,Ny/2+1)][1] + dt/2.0*(3*VWz_N[I3(i,j,l,Ny/2+1)][1] - VWz_N_1[I3(i,j,l,Ny/2+1)][1]) + nu*dt/2.0*Vz_N[I3(i,j,l,Ny/2+1)][1];
      }
    }
  }
  // Now our V^N contains V^(N+1/3) + nu*dt/2* laplacian (V^N)
  
  // Now its time to build Pi_hat
  build_pi_hat(Vx_N,Vy_N,Vz_N,D2,PI,temp1);
 

  build_rhs(temp1,temp2,temp3,temp4,Vx_N,Vy_N,Vz_N,PI,D2,temp_rhs);
  // up to now, Pi is Pi^ 
  // temp1 =  phii^-1 ( Vx^(N+1/3) + nu*dt/2 laplacian Vx^N - gradx (pi hat))
  // temp2 =  phii^-1 ( Vy^(N+1/3) + nu*dt/2 laplacian Vy^N - grady (pi hat))
  // temp3 =  phii^-1 ( Vz^(N+1/3) + nu*dt/2 laplacian Vz^N - gradz (pi hat))
  // temp4 =  Dz(PI^)
  // we will need Dz(PI^), M-1 element which is Nz-2
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      Dz_phi_hat_M_1[I(i,j,Ny/2+1)][0] = temp4[I3(i,j,Nz-2,Ny/2+1)][0];
      Dz_phi_hat_M_1[I(i,j,Ny/2+1)][1] = temp4[I3(i,j,Nz-2,Ny/2+1)][1];
    }
  }

  Dz(temp3,temp4);
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      if( i!=0 | j !=0){
	// copy tau matrix that we built into TAU matrix 
	for(int l=0;l<4;l++)
	  TAU(l,0) = c_tau1[I3(i,j,l,Ny/2+1)];
	for(int l=0;l<4;l++)
	  TAU(l,1) = c_tau2[I3(i,j,l,Ny/2+1)];
	for(int l=0;l<4;l++)
	  TAU(l,2) = c_tau7[I3(i,j,l,Ny/2+1)];
	for(int l=0;l<4;l++)
	  TAU(l,3) = c_tau8[I3(i,j,l,Ny/2+1)]; 
	//RHS[0] = - (Vz_N[I3(i,j,Nz-1,Ny/2+1)][0]);
	//RHS[1] = Vz_N[I3(i,j,Nz-2,Ny/2+1)][0] -Dz_phi_hat_M_1[I(i,j,Ny/2+1)][0] ;
	RHS[0] = temp3[I3(i,j,Nz-1,Ny/2+1)][0] - (temp_rhs[I3(i,j,0,Ny/2+1)][0])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	RHS[1] = temp3[I3(i,j,Nz-2,Ny/2+1)][0] - (temp_rhs[I3(i,j,1,Ny/2+1)][0])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	RHS[2] = -Kx[i]*temp1[I3(i,j,Nz-1,Ny/2+1)][1] - Ky[j]*temp2[I3(i,j,Nz-1,Ny/2+1)][1] + temp4[I3(i,j,Nz-1,Ny/2+1)][0];
	RHS[3] = -Kx[i]*temp1[I3(i,j,Nz-2,Ny/2+1)][1] - Ky[j]*temp2[I3(i,j,Nz-2,Ny/2+1)][1] + temp4[I3(i,j,Nz-2,Ny/2+1)][0]; 
	//cout<<"i :"<<i<<"\t j :"<<j<<"\n";

	//cout<<c_tau1[I3(i,j,0,Ny/2+1)]<<"\t"<<c_tau1[I3(i,j,1,Ny/2+1)]<<"\t"<<c_tau1[I3(i,j,2,Ny/2+1)]<<"\t"<<c_tau1[I3(i,j,3,Ny/2+1)]<<endl;
	//cout<<"tau is :\n"<<TAU;
	solve(SOL,TAU,RHS);
	//cout<<"RHS is:\n"<<RHS;
	//cout<<"SOL is :\n"<<SOL;
	tau1[0] = SOL(0);
	tau2[0] = SOL(1);
	tau7[0] = SOL(2);
	tau8[0] = SOL(3);
	
	//RHS[0] = - (Vz_N[I3(i,j,Nz-1,Ny/2+1)][1]);
	//RHS[1] = Vz_N[I3(i,j,Nz-2,Ny/2+1)][1] - Dz_phi_hat_M_1[I(i,j,Ny/2+1)][0]; // gradz of PI^ in here is missing 
	RHS[0] = temp3[I3(i,j,Nz-1,Ny/2+1)][1] - (temp_rhs[I3(i,j,0,Ny/2+1)][1])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j]));
	RHS[1] = temp3[I3(i,j,Nz-2,Ny/2+1)][1] - (temp_rhs[I3(i,j,1,Ny/2+1)][1])/(1.0 + nu*dt/2.0*(Kx[i]*Kx[i] + Ky[j]*Ky[j])); 
	RHS[2] = Kx[i]*temp1[I3(i,j,Nz-1,Ny/2+1)][0] + Ky[j]*temp2[I3(i,j,Nz-1,Ny/2+1)][0] + temp4[I3(i,j,Nz-1,Ny/2+1)][1];
	RHS[3] = Kx[i]*temp1[I3(i,j,Nz-2,Ny/2+1)][0] + Ky[j]*temp2[I3(i,j,Nz-2,Ny/2+1)][0] + temp4[I3(i,j,Nz-2,Ny/2+1)][1]; 
	//cout<<"i :"<<i<<"\t j :"<<j<<"\n";
	//cout<<TAU;
	solve(SOL,TAU,RHS);
	// cout<<SOL;
	//cout<<"RHS is:\n"<<RHS;
	//cout<<"SOL is :\n"<<SOL;
	tau1[1] = SOL(0);
	tau2[1] = SOL(1);
	tau7[1] = SOL(2);
	tau8[1] = SOL(3);

	for(int l=0;l<Nz;l++){
	  tempVx[I3(i,j,l,Ny/2+1)][0] = temp1[I3(i,j,l,Ny/2+1)][0]  + tau1[1] * phi_inv_grad_G1x[I3(i,j,l,Ny/2+1)][1]  + tau2[1] * phi_inv_grad_G2x[I3(i,j,l,Ny/2+1)][1]  + tau7[1] * phi_inv_grad_G7x[I3(i,j,l,Ny/2+1)][1]  + tau8[1] * phi_inv_grad_G8x[I3(i,j,l,Ny/2+1)][1];

	  tempVx[I3(i,j,l,Ny/2+1)][1] = temp1[I3(i,j,l,Ny/2+1)][1] - tau1[0] * phi_inv_grad_G1x[I3(i,j,l,Ny/2+1)][1]  - tau2[0] * phi_inv_grad_G2x[I3(i,j,l,Ny/2+1)][1]  - tau7[0] * phi_inv_grad_G7x[I3(i,j,l,Ny/2+1)][1]  - tau8[0] * phi_inv_grad_G8x[I3(i,j,l,Ny/2+1)][1] ;
	

	  tempVy[I3(i,j,l,Ny/2+1)][0] = temp2[I3(i,j,l,Ny/2+1)][0]  + tau1[1] * phi_inv_grad_G1y[I3(i,j,l,Ny/2+1)][1]  + tau2[1] * phi_inv_grad_G2y[I3(i,j,l,Ny/2+1)][1]  + tau7[1] * phi_inv_grad_G7y[I3(i,j,l,Ny/2+1)][1]  + tau8[1] * phi_inv_grad_G8y[I3(i,j,l,Ny/2+1)][1];

	  tempVy[I3(i,j,l,Ny/2+1)][1] = temp2[I3(i,j,l,Ny/2+1)][1] - tau1[0] * phi_inv_grad_G1y[I3(i,j,l,Ny/2+1)][1]  - tau2[0] * phi_inv_grad_G2y[I3(i,j,l,Ny/2+1)][1]  - tau7[0] * phi_inv_grad_G7y[I3(i,j,l,Ny/2+1)][1]  - tau8[0] * phi_inv_grad_G8y[I3(i,j,l,Ny/2+1)][1] ;

	  tempVz[I3(i,j,l,Ny/2+1)][0] = temp3[I3(i,j,l,Ny/2+1)][0] - tau1[0] * phi_inv_grad_G1z[I3(i,j,l,Ny/2+1)][0]  - tau2[0] * phi_inv_grad_G2z[I3(i,j,l,Ny/2+1)][0]  - tau7[0] * phi_inv_grad_G7z[I3(i,j,l,Ny/2+1)][0]  - tau8[0] * phi_inv_grad_G8z[I3(i,j,l,Ny/2+1)][0] ;

	  tempVz[I3(i,j,l,Ny/2+1)][1] = temp3[I3(i,j,l,Ny/2+1)][1]  - tau1[1] * phi_inv_grad_G1z[I3(i,j,l,Ny/2+1)][0]  - tau2[1] * phi_inv_grad_G2z[I3(i,j,l,Ny/2+1)][0]  - tau7[1] * phi_inv_grad_G7z[I3(i,j,l,Ny/2+1)][0]  - tau8[1] * phi_inv_grad_G8z[I3(i,j,l,Ny/2+1)][0];

	}

      }  
    }
  }
  //solving the zero zero mode 
  zero_zero_mode_solve(temp1, temp2, temp3, Vx_N, Vy_N, Vz_N, D2);
  for(int l=0;l<Nz;l++){
    tempVz[I3(0,0,l,Ny/2+1)][0] = 0.0;//temp3[I3(0,0,l,Ny/2+1)][0]; //0.0;
    tempVz[I3(0,0,l,Ny/2+1)][1] = 0.0;//;temp3[I3(0,0,l,Ny/2+1)][1];//0.0;
    
    tempVx[I3(0,0,l,Ny/2+1)][0] = temp1[I3(0,0,l,Ny/2+1)][0]; // Vx_N[I3(0,0,l,Ny/2+1)][0];
    tempVx[I3(0,0,l,Ny/2+1)][1] = temp1[I3(0,0,l,Ny/2+1)][1]; //Vx_N[I3(0,0,l,Ny/2+1)][1];
  
    tempVy[I3(0,0,l,Ny/2+1)][0] = temp2[I3(0,0,l,Ny/2+1)][0]; //Vy_N[I3(0,0,l,Ny/2+1)][0];
    tempVy[I3(0,0,l,Ny/2+1)][1] = temp2[I3(0,0,l,Ny/2+1)][1]; //Vy_N[I3(0,0,l,Ny/2+1)][1];
  }
  
  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	Vx_N_1[I3(i,j,l,Ny/2+1)][0] = tempVx[I3(i,j,l,Ny/2+1)][0];
	Vx_N_1[I3(i,j,l,Ny/2+1)][1] = tempVx[I3(i,j,l,Ny/2+1)][1];

	Vy_N_1[I3(i,j,l,Ny/2+1)][0] = tempVy[I3(i,j,l,Ny/2+1)][0];
	Vy_N_1[I3(i,j,l,Ny/2+1)][1] = tempVy[I3(i,j,l,Ny/2+1)][1];

	Vz_N_1[I3(i,j,l,Ny/2+1)][0] = tempVz[I3(i,j,l,Ny/2+1)][0];
	Vz_N_1[I3(i,j,l,Ny/2+1)][1] = tempVz[I3(i,j,l,Ny/2+1)][1];
      }
    }
  }
  

  CFF_transformer_backward(d_Vx_N_1,Vx_N_1);
  CFF_transformer_backward(d_Vy_N_1,Vy_N_1);
  CFF_transformer_backward(d_Vz_N_1,Vz_N_1);

  omega_calculator(d_Wx,d_Wy,d_Wz,d_temp1,d_temp2,Vx_N_1,Vy_N_1,Vz_N_1,temp1,temp2);
  cross_product(d_Vx_N_1, d_Vy_N_1, d_Vz_N_1, d_Wx, d_Wy, d_Wz, d_VWx, d_VWy, d_VWz, temp1, temp2, temp3);

  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	VWx_N_1[I3(i,j,l,Ny/2+1)][0] = VWx_N[I3(i,j,l,Ny/2+1)][0];
	VWx_N_1[I3(i,j,l,Ny/2+1)][1] = VWx_N[I3(i,j,l,Ny/2+1)][1];

	VWy_N_1[I3(i,j,l,Ny/2+1)][0] = VWy_N[I3(i,j,l,Ny/2+1)][0];
	VWy_N_1[I3(i,j,l,Ny/2+1)][1] = VWy_N[I3(i,j,l,Ny/2+1)][1];

	VWz_N_1[I3(i,j,l,Ny/2+1)][0] = VWz_N[I3(i,j,l,Ny/2+1)][0];
	VWz_N_1[I3(i,j,l,Ny/2+1)][1] = VWz_N[I3(i,j,l,Ny/2+1)][1];
	
	VWx_N[I3(i,j,l,Ny/2+1)][0] = temp1[I3(i,j,l,Ny/2+1)][0];
	VWx_N[I3(i,j,l,Ny/2+1)][1] = temp1[I3(i,j,l,Ny/2+1)][1];

	VWy_N[I3(i,j,l,Ny/2+1)][0] = temp2[I3(i,j,l,Ny/2+1)][0];
	VWy_N[I3(i,j,l,Ny/2+1)][1] = temp2[I3(i,j,l,Ny/2+1)][1];

	VWz_N[I3(i,j,l,Ny/2+1)][0] = temp3[I3(i,j,l,Ny/2+1)][0];
	VWz_N[I3(i,j,l,Ny/2+1)][1] = temp3[I3(i,j,l,Ny/2+1)][1];
	
      }
    }
  }
  time = time + dt;
  cout<<"t = "<<time<<endl;
  //********************************************************************************
  //computing to see if the field is divergence free 
  
  energy = 0.0;
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	energy = energy + 0.5*( d_Vx_N_1[I3(i,j,l,Ny)]*d_Vx_N_1[I3(i,j,l,Ny)] + d_Vy_N_1[I3(i,j,l,Ny)]*d_Vy_N_1[I3(i,j,l,Ny)] + d_Vz_N_1[I3(i,j,l,Ny)]*d_Vz_N_1[I3(i,j,l,Ny)] );
      }
    }
  }
  out2<<time<<"\t"<<energy<<"\n";

  out4<<time<<"\t"<<d_Vz_N_1[I3(Nx/2,Ny/2,Nz-1,Ny)]<<"\t"<<d_Vz_N_1[I3(Nx/2,Ny/2,0,Ny)]<<"\n";
  

  if(int(time/dt)%100 ==0 ){
    for(int l=0;l<Nz;l++){
      zz= cos(M_PI*(double)l/(double)(Nz-1));
      out3<<zz<<"\t"<<d_Vx_N_1[I3(Nx/2,Ny/2,l,Ny)]<<endl;
    }
    out3<<"\n\n\n";
  }
  
  Dz(Vz_N_1,temp1);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      for(int l=0;l<Nz;l++){
	temp2[I3(i,j,l,Ny/2+1)][0] = - Kx[i]*Vx_N_1[I3(i,j,l,Ny/2+1)][1] - Ky[j]*Vy_N_1[I3(i,j,l,Ny/2+1)][1] + temp1[I3(i,j,l,Ny/2+1)][0];
	temp2[I3(i,j,l,Ny/2+1)][1] =   Kx[i]*Vx_N_1[I3(i,j,l,Ny/2+1)][0] + Ky[j]*Vy_N_1[I3(i,j,l,Ny/2+1)][0] + temp1[I3(i,j,l,Ny/2+1)][1];
      }
    }
  }    
  CFF_transformer_backward(d_temp2, temp2);
  //  for(int i=0;i<Nx;i++)
  //  for(int j=0;j<Ny;j++)
  //  for(int l=0;l<Nz;l++){
  out1<<time<<"\t"<<norm_infinity(d_temp2)<<"\n";
  //}

  

  // up to here for computing and giving the divergence    
  //********************************************************************************
  

 

  /*
  for(int l=0;l<Nz;l++)
    cout<<d_Vx_N_1[I3(2,2,l,Ny)]<<endl;//<<"\t"<<d_Vy_N[I3(0,0,l,Ny)]<<"\n";//<<d_Vx_N[I3(i,j,0,Ny)]<<"\n";
  cout<<"\n\n\n";
  */
  /*
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++)
	cout<<d_Vx_N[I3(i,j,Nz-1,Ny)]<<"\t"<<d_Vx_N[I3(i,j,0,Ny)]<<"\n";
      //cout<<"\n\n\n";
    }
  }
  */



  }
  
  /*
    //Cheching omega perpendicular
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++){
	cout<<d_Vx_N[I3(i,j,l,Ny)]*d_VWx[I3(i,j,l,Ny)]+d_Vz_N[I3(i,j,l,Ny)]*d_VWz[I3(i,j,l,Ny)]+d_Vz_N[I3(i,j,l,Ny)]*d_VWz[I3(i,j,l,Ny)]<<endl;
      }
    }
  }




   */  


  /*
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      //for(int l=0;l<Nz;l++)
      cout<<d_Vx_N[I3(i,j,N-1,Ny)]<<"\t"<<d_Vx_N[I3(i,j,0,Ny)]"\n";
      cout<<"\n\n\n";
    }
  }

  */

  /*
  cout<<"\n\n\n";
  cout<<"d_G2 is:"<<endl;
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      cout<<d_G2[I3(i,j,Nz-2,Ny/2+1)]<<"\t"<<d_G2[I3(i,j,Nz-1,Ny/2+1)]<<endl;
    }
  }
  

  cout<<"\n\n\n";
  cout<<"dTm/dz is:"<<endl;
  for(int l=0;l<Nz;l++){
    cout<<dTm[l]<<"\n";
  }

  cout<<"\n\n\n";
  cout<<"tau coeff matrix is:"<<endl;
  for(int l=0;l<4;l++)
    cout<<c_tau1[I3(1,1,l,Ny/2+1)]<<"\t"<<c_tau2[I3(1,1,l,Ny/2+1)]<<"\t"<<c_tau7[I3(1,1,l,Ny/2+1)]<<"\t"<<c_tau8[I3(1,1,l,Ny/2+1)]<<endl;
  */

  /*
  CFF_transformer_backward(d_Vx_N,Vx_N);
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++)
	cout<<d_Vz_N[I3(i,j,l,Ny)]<<"\n";
      cout<<"\n\n\n";
    }
  }
  */


  /*
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int l=0;l<Nz;l++)
	cout<<d_Wz[I3(i,j,l,Ny)]<<"\n";
      cout<<"\n\n\n";
    }
  }  
  for(int i=0;i<Nx;i++){
    for(int j=0;j<Ny/2+1;j++){
      if(i!=0 & j!=0)
	for(int l=0;l<Nz;l++)
	cout<<d_G8[I3(i,j,l,Ny/2+1)]<<"\n";
    cout<<"\n\n\n";
    }
  }
  */


  
  /*
  for(int i=0;i<Nz;i++){
    for(int j=0;j<Nz;j++){
      cout<<D2[I(i,j,Nz)]<<"\t";
    }
    cout<<"\n";
  }
  */
  
  
  fftw_free(D2);
  fftw_free(d_Vx_N_1);
  fftw_free(d_Vy_N_1);
  fftw_free(d_Vz_N_1);
  fftw_free(d_Vx_N);
  fftw_free(d_Vy_N);
  fftw_free(d_Vz_N);
  fftw_free(d_Wx);
  fftw_free(d_Wy);
  fftw_free(d_Wz);
  fftw_free(d_temp1);
  fftw_free(d_temp2);
  fftw_free(temp1);
  fftw_free(temp2);
  fftw_free(Vx_N_1);
  fftw_free(Vy_N_1);
  fftw_free(Vz_N_1);
  fftw_free(Vx_N);
  fftw_free(Vy_N);
  fftw_free(Vz_N);
  //  fftw_free(Wx);
  //  fftw_free(Wy);
  //  fftw_free(Wz);
  fftw_free(VWx_N_1);
  fftw_free(VWy_N_1);
  fftw_free(VWz_N_1);
  fftw_free(VWx_N);
  fftw_free(VWy_N);
  fftw_free(VWz_N);
  fftw_free(d_G1);
  fftw_free(d_G2);
  fftw_free(d_G7);
  fftw_free(d_G8);
  fftw_free(d_PI);
  fftw_free(PI);
  fftw_free(c_tau1);
  fftw_free(c_tau2);
  fftw_free(c_tau7);
  fftw_free(c_tau8);

 return 0;
}
