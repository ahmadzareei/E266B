#include<iostream>
#include<fstream>
#include<fftw3.h>
#include<cmath>
#include<cstdlib>

using namespace std;

void Chebyshev_transformer(int N, double * in, double *out, int forward){
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

int main(){

  ofstream out1,out2;
  int N=100;
  double *in;
  double *out;
  double x,theta;
  int forward;

  out1.open("data.dat");
  out2.open("BF_data.dat");


  in = fftw_alloc_real(sizeof(double)*N);
  out = fftw_alloc_real(sizeof(double)*N);
  

  for(int i=0;i<N;i++){
    x= cos(M_PI/(double)(N-1)*(double)i);
    in[i] = 3.*x*x-3.;
    out1<<x<<"\t"<<in[i]<<"\n";
  }

  forward = 1;
  Chebyshev_transformer(N, in, out, forward);
 
  forward =-1;
  Chebyshev_transformer(N, out, in, forward);

  for(int i=0;i<N;i++){
    x= cos(M_PI/(double)(N-1)*(double)i);
    out2<<x<<"\t"<<in[i]<<"\n";
  }

  
  out1.close();
  out2.close();
  fftw_free(in);
  fftw_free(out);

  return 0;
}



/************************************************/
