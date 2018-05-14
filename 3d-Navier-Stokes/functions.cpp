

int I3(int i, int j, int l,int N);
inline int I(int i, int j,int N);
void chebyshev_transformer(int N, double * in, double *out, int forward);
void fft_forward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V);
void fft_backward(fftw_plan plan, int Nx, int Ny, double * v, fftw_complex * V);
void zero_out( fftw_complex * V,int Nx, int Ny);
void chebyshev_derivative(int N, double * in, double * out);
void papper_cutter(int N, double * A, double *X, double *Y);
void CFF_transformer_forward(double * d_phi, fftw_complex *phi);
void CFF_transformer_backward(double * d_phi, fftw_complex *phi);
void Ai(int N, double *A, int index);
void build_D2(int N, double *D2);
void build_G1278(double *D2,double *G1,double *G2,double *G7,double *G8);
void omega_calculator(double *d_Wx,double *d_Wy,double *d_Wz,double *d_temp1,double *d_temp2,fftw_complex *Vx,fftw_complex *Vy,fftw_complex *Vz,fftw_complex *temp1,fftw_complex *temp2);
void Dz(fftw_complex *in, fftw_complex *out);
void cross_product(double *d_Vx,double *d_Vy,double *d_Vz, double *d_Wx, double *d_Wy, double *d_Wz, double *d_VWx, double *d_VWy, double *d_VWz,fftw_complex  *VWx, fftw_complex *VWy, VWz);
void laplacian(fftw_complex *in, fftw_complex *out);
