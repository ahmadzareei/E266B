#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>

using namespace std;
/**************************************************************************************/
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
/**************************************************************************************/
void chebyshev_integrator_1D(int N, double *f, double * g, double boundary, double boundary_value){
  // it solves df(x)/dx = g(x), and solve for f_i 
  // supposing boundary condition f(x=boundary) = boundary_value
  //boundary could only be +1 or -1

  double sum = 0.0;
  double sign = boundary;

  //last two rows
  f[N-1] = g[N-2] / (2.0*(double)(N-1));
  f[N-2] = f[N-3] / (2.0*(double)(N-2));

  //middle section rows
  for(int i= N-4;i>0;i--){
    sum = 0.0;
    for(int j = i+3; j<N;j=j+2){
      sum = sum + 2.0*(double)j*f[j];
    }
    f[i+1] = (g[i] - sum)/(2.0*(double)(i+1));
  }
  
  // the first row of the matrix
  sum = 0.0;
  for(int i = 3;i<N;i++){
    sum = sum + (double)(i)*f[i];
  }
  f[1] = g[0] - sum ;

  // The last row that has been removed and the boudnary condition subistutued!
  sum=0.0;
  for(int i=1;i<N;i++){
    sum = sum + sign * f[i];
    sign = sign * boundary;
  }

  f[0] = boundary_value-sum;

}
/*************************************************************************************************/
int main(){

  ofstream out1,out2;
  int N=100;
  double *in_phys;
  double *out_chebysh;
  double x,theta;
  int forward;
  
  double boundary = 1.0;
  double boundary_value = 1.0;

  out1.open("data.dat");
  out2.open("BF_data.dat");


  in_phys = fftw_alloc_real(sizeof(double)*N);
  out_chebysh = fftw_alloc_real(sizeof(double)*N);
  

  for(int i=0;i<N;i++){
    x= cos(M_PI/(double)(N-1)*(double)i);
    in_phys[i] = 3.*x*x-3.;
    out1<<x<<"\t"<<in_phys[i]<<"\t"<<x*x*x-3.0*x+3.0<<"\n";
  }

  forward = 1;
  chebyshev_transformer(N, in_phys, out_chebysh, forward);
  

  chebyshev_integrator_1D(N,in_phys,out_chebysh,boundary, boundary_value);

  for(int i=0;i<N;i++)
    out_chebysh[i]= in_phys[i];
 
  forward =-1;
  chebyshev_transformer(N, out_chebysh, in_phys, forward);

  for(int i=0;i<N;i++){
    x= cos(M_PI/(double)(N-1)*(double)i);
    out2<<x<<"\t"<<in_phys[i]<<"\n";
  }

  
  out1.close();
  out2.close();
  fftw_free(in_phys);
  fftw_free(out_chebysh);

  return 0;
}




