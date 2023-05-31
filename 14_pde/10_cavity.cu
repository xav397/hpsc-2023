#include <cstdio>
#include <sys/stat.h>

struct Parameter {
    int nx;
    int ny;
    int size;
    int nit;
    double dx;
    double dy;
    double dt;
    double rho;
    double nu;

    Parameter( ) {
        this->nx = 41;
        this->ny = 41;
        this->size = this->nx * this->ny;
        this->nit = 50;
        this->dx = 2. / ((double)(this->nx) - 1.);
        this->dy = 2. / ((double)(this->ny) - 1.);
        this->dt = .01;
        this->rho = 1.;
        this->nu = .02;
    }
};

__global__
void calc_b
(
    Parameter param,
    double *b,
    double *u,
    double *v
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= param.size ) return;
    const int nx = param.nx;
    const int ny = param.ny;
    const double dx = param.dx;
    const double dy = param.dy;
    const double dt = param.dt;
    const double rho = param.rho;
    int i = idx % nx;
    int j = idx / nx;

    if(i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        b[nx*j+i] = rho * (1/dt *
                  ((u[nx*j+i+1] - u[nx*j+i-1]) / (2. * dx) + (v[nx*(j+1)+i] - v[nx*(j-1)+i]) / (2. * dy)) -\
               pow((u[nx*j+i+1] - u[nx*j+i-1]) / (2. * dx), 2) - 2. * ((u[nx*(j+1)+i] - u[nx*(j-1)+i]) / (2. * dy) *\
                   (v[nx*j+i+1] - v[nx*j+i-1]) / (2. * dx)) - pow((v[nx*(j+1)+i] - v[nx*(j-1)+i]) / (2. * dy), 2));
    }
}

__global__
void calc_p
(
    Parameter param,
    double *b,
    double *p,
    double *pn
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= param.size ) return;
    const int nx = param.nx;
    const int ny = param.ny;
    const double dx = param.dx;
    const double dy = param.dy;
    int i = idx % nx;
    int j = idx / nx;

    if(i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        p[idx] = (pow(dy, 2) * (pn[nx*j+i+1]   + pn[nx*j+i-1])   +\
                  pow(dx, 2) * (pn[nx*(j+1)+i] + pn[nx*(j-1)+i]) -\
                  b[nx*j+i] * pow(dx, 2) * pow(dy, 2))\
                 / (2. * (pow(dx, 2) + pow(dy, 2)));
    }
}

__global__
void apply_boundary_p
(
    Parameter param,
    double *p
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= param.size ) return;
    const int nx = param.nx;
    const int ny = param.ny;
    int i = idx % nx;
    int j = idx / nx;

    if(i == nx-1) p[idx] = p[nx*j+i-1];
    if(j == 0)    p[idx] = p[nx*(j+1)+i];
    if(i == 0)    p[idx] = p[nx*j+i+1];
    if(j == ny-1) p[idx] = 0.;
}

__global__
void calc_velocity
(
    Parameter param,
    double *u,
    double *v,
    double *p,
    double *un,
    double *vn
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= param.size ) return;
    const int nx = param.nx;
    const int ny = param.ny;
    const double dx = param.dx;
    const double dy = param.dy;
    const double dt = param.dt;
    const double rho = param.rho;
    const double nu = param.nu;
    int i = idx % nx;
    int j = idx / nx;

    if(i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        u[idx] = un[nx*j+i] - un[nx*j+i] * dt / dx * (un[nx*j+i] - un[nx*j+i - 1])\
                            - un[nx*j+i] * dt / dy * (un[nx*j+i] - un[nx*(j-1)+i])\
                            - dt / (2. * rho * dx) * (p[nx*j+i+1] - p[nx*j+i-1])\
                            + nu * dt / pow(dx, 2) * (un[nx*j+i+1] - 2. * un[nx*j+i] + un[nx*j+i-1])\
                            + nu * dt / pow(dy, 2) * (un[nx*(j+1)+i] - 2. * un[nx*j+i] + un[nx*(j-1)+i]);
        v[idx] = vn[nx*j+i] - vn[nx*j+i] * dt / dx * (vn[nx*j+i] - vn[nx*j+i - 1])\
                            - vn[nx*j+i] * dt / dy * (vn[nx*j+i] - vn[nx*(j-1)+i])\
                            - dt / (2. * rho * dx) * (p[nx*(j+1)+i] - p[nx*(j-1)+i])\
                            + nu * dt / pow(dx, 2) * (vn[nx*j+i+1] - 2. * vn[nx*j+i] + vn[nx*j+i-1])\
                            + nu * dt / pow(dy, 2) * (vn[nx*(j+1)+i] - 2. * vn[nx*j+i] + vn[nx*(j-1)+i]);
    }
}

__global__
void apply_boundary_velocity
(
    Parameter param,
    double *u,
    double *v
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= param.size ) return;
    const int nx = param.nx;
    const int ny = param.ny;
    int i = idx % nx;
    int j = idx / nx;

    if(j == 0) {
        u[idx] = 0.;
        v[idx] = 0.;
    }
    if(i == 0) {
        u[idx] = 0.;
        v[idx] = 0.;
    }
    if(i == nx-1) {
        u[idx] = 0.;
        v[idx] = 0.;
    }
    if(j == ny-1) {
        u[idx] = 1.;
        v[idx] = 0.;
    }
}

__host__
void output
(
    const int step,
    const Parameter param,
    const double *u,
    const double *v,
    const double *p
);

int main() {
    Parameter param;
    const int nt = 500;

    double *x, *y, *u, *v, *p, *b, *pn, *un, *vn;
    const int size = param.size;
    cudaMallocManaged(&x, size*sizeof(double));
    cudaMallocManaged(&y, size*sizeof(double));
    cudaMallocManaged(&u, size*sizeof(double));
    cudaMallocManaged(&v, size*sizeof(double));
    cudaMallocManaged(&p, size*sizeof(double));
    cudaMallocManaged(&b, size*sizeof(double));
    cudaMallocManaged(&pn, size*sizeof(double));
    cudaMallocManaged(&un, size*sizeof(double));
    cudaMallocManaged(&vn, size*sizeof(double));

    cudaMemset(u, 0., size*sizeof(double));
    cudaMemset(v, 0., size*sizeof(double));
    cudaMemset(p, 0., size*sizeof(double));
    cudaMemset(b, 0., size*sizeof(double));

    const int thread_num = 1024;
    const int grid_x = (size+thread_num-1)/thread_num;

    for(int n = 0; n < nt; n++){
        calc_b<<<grid_x,thread_num>>>(param, b, u, v);
        cudaDeviceSynchronize();
        
        for(int it = 0; it < param.nit; it++) {
            cudaMemcpy(pn, p, size*sizeof(double), cudaMemcpyDeviceToDevice);
            
            calc_p<<<grid_x,thread_num>>>(param, b, p, pn);
            cudaDeviceSynchronize();

            apply_boundary_p<<<grid_x,thread_num>>>(param, p);
            cudaDeviceSynchronize();
        }
        
        cudaMemcpy(un, u, size*sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(vn, v, size*sizeof(double), cudaMemcpyDeviceToDevice);

        calc_velocity<<<grid_x,thread_num>>>(param, u, v, p, un, vn);
        cudaDeviceSynchronize();

        apply_boundary_velocity<<<grid_x,thread_num>>>(param, u, v);
        cudaDeviceSynchronize();

        //output(n, param, u, v, p); //output files for visualization
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(pn);
    cudaFree(un);
    cudaFree(vn);

    return 0;
}


__host__
void output
(
    const int step,
    const Parameter param,
    const double *u,
    const double *v,
    const double *p
)
{
    char dirname[1256] = "./result";
    if(step == 0) mkdir(dirname, S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH );

    FILE *pFile;
    char str[100];
    int i, j;
    const int nx = param.nx;
    const int ny = param.ny;
    sprintf(str, "%s/plot-%.4d.vtu",dirname,step);
    pFile = fopen(str, "w");

    fprintf(pFile, "<VTKFile type = \"UnstructuredGrid\" version = \"0.1\" byte_order = \"BigEndian\" header_type=\"UInt64\">\n");
  
    fprintf(pFile, "<UnstructuredGrid>\n");

    fprintf(pFile, "<Piece NumberOfPoints = \"%d\" NumberOfCells = \"%d\">\n", param.size, (param.nx-1)*(param.ny-1));

    fprintf(pFile, "<PointData>\n");

    fprintf(pFile, "<DataArray type = \"Float32\" Name = \"velocity\" NumberOfComponents = \"3\" Format = \"ascii\">\n");
    for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
	    fprintf(pFile, "%.3e %.3e %.3e\n", u[nx*j+i], v[nx*j+i], 0.);
    }
    }
    fprintf(pFile, "</DataArray>\n");

    fprintf(pFile, "<DataArray type = \"Float32\" Name = \"pressure\" NumberOfComponents = \"1\" Format = \"ascii\">\n");
    for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
        fprintf(pFile, "%.3e\n", p[nx*j+i]);
    }
    }
    fprintf(pFile, "</DataArray>\n");

    fprintf(pFile, "</PointData>\n");
	fprintf(pFile, "<Points>\n");

	fprintf(pFile, "<DataArray type = \"Float32\" NumberOfComponents = \"3\" Format = \"ascii\">\n");
	for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
    	fprintf(pFile, "%.5e %.5e %.5e\n", param.dx*i, param.dy*j, 0.);
	}
	}
	fprintf(pFile, "</DataArray>\n");

	fprintf(pFile, "</Points>\n");
	fprintf(pFile, "<Cells>\n");

	fprintf(pFile, "<DataArray type = \"Int32\" Name = \"connectivity\" Format = \"ascii\">\n");
	for (j = 0; j < ny-1; j++) {
	for (i = 0; i < nx-1; i++) {
		fprintf(pFile, "%d %d %d %d\n", nx*j+i, nx*j+i+1, nx*(j+1)+i+1, nx*(j+1)+i);
	}
	}
	fprintf(pFile, "</DataArray>\n");

	fprintf(pFile, "<DataArray type = \"Int32\" Name = \"offsets\" Format = \"ascii\">\n");
	for (i = 1; i <= nx*ny; i++) {
		fprintf(pFile, "%d ", 4 * i);
	}
	fprintf(pFile, "\n</DataArray>\n");

	fprintf(pFile, "<DataArray type = \"Int32\" Name = \"types\" Format = \"ascii\">\n");
	for (i = 1; i <= nx*ny; i++) {
		fprintf(pFile, "10 ");
	}
	fprintf(pFile, "\n</DataArray>\n");

	fprintf(pFile, "</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>");
	fclose(pFile);
}