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
/*****************************************************************************************/
void chebyshev_derivative(int N, double * in, double * out){

  out[N-1]= 0.0;
  out[N-2] = 2.0*(double)(N-1)*in[N-1];
  
  for(int k=N-3;k>0;k-=1){
    out[k] = 2*(double)(k+1)*in[k+1] + out[k+2];
  }
  out[0] = in[1] + out[2]/2.0;

}
/*****************************************************************************************/

int main(){

  ofstream out1,out2;
  int N=100;
  double *in_phys;
  double *out_chebysh;
  double x,theta;
  int forward;

  out1.open("data.dat");
  out2.open("BF_data.dat");


  in_phys = fftw_alloc_real(sizeof(double)*N);
  out_chebysh = fftw_alloc_real(sizeof(double)*N);
  

  for(int i=0;i<N;i++){
    x= cos(M_PI/(double)(N-1)*(double)i);
    in_phys[i] = 3.*x*x-3.;
    out1<<x<<"\t"<<in_phys[i]<<"\n";
  }

  forward = 1;
  chebyshev_transformer(N, in_phys, out_chebysh, forward);

  chebyshev_derivative(N,out_chebysh,in_phys);

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



/************************************************/
