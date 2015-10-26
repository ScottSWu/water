#include <omp.h>
#include <offload.h>

#define ALLOC alloc_if(1)
#define FREE free_if(1)
#define RETAIN free_if(0)
#define REUSE alloc_if(0)


#ifndef CENTRAL2D_H
#define CENTRAL2D_H


#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include "minmod.h"
#include "immintrin.h"

#ifndef NX  
#define NX 200
#endif
 
#ifndef NX_ALL
#define NX_ALL 206
#endif

float *u1_;             // Solution values
float *u2_;             
float *u3_;  
float *f1_;             // Fluxes in x
float *f2_; 
float *f3_;  
float *g1_;             // Fluxes in y
float *g2_; 
float *g3_; 
float *ux1_;           // x differences of u
float *ux2_;
float *ux3_;
float *uy1_;           // y differences of u
float *uy2_;
float *uy3_;
float *fx1_;           // x differences of f
float *fx2_;
float *fx3_;
float *gy1_;           // y differences of g
float *gy2_;
float *gy3_;
float *v1_;             // Solution values at next step
float *v2_; 
float *v3_; 


class Central2D {
public:

    Central2D(float w, float h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              float cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny),
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(w/nx), dy(h/ny),
        cfl(cfl) {}
    
    static constexpr int nghost = 3;   // Number of ghost cells
    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const float dx, dy;         // Cell size in x/y
    const float cfl;            // Allowed CFL number
    // Array accessor functions
    int offset(int ix, int iy) const { return iy*nx_all+ix; }

    float& u1(int ix, int iy) { return u1_[offset(ix,iy)]; }            // Solution values
    float& u2(int ix, int iy) { return u2_[offset(ix,iy)]; }            
    float& u3(int ix, int iy) { return u3_[offset(ix,iy)]; } 
    float& f1(int ix, int iy) { return f1_[offset(ix,iy)]; }            // Fluxes in x
    float& f2(int ix, int iy) { return f2_[offset(ix,iy)]; }
    float& f3(int ix, int iy) { return f3_[offset(ix,iy)]; } 
    float& g1(int ix, int iy) { return g1_[offset(ix,iy)]; }            // Fluxes in y
    float& g2(int ix, int iy) { return g2_[offset(ix,iy)]; }
    float& g3(int ix, int iy) { return g3_[offset(ix,iy)]; }
    float& ux1(int ix, int iy) { return ux1_[offset(ix,iy)]; }           // x differences of u
    float& ux2(int ix, int iy) { return ux2_[offset(ix,iy)]; }
    float& ux3(int ix, int iy) { return ux3_[offset(ix,iy)]; }
    float& uy1(int ix, int iy) { return uy1_[offset(ix,iy)]; }           // y differences of u
    float& uy2(int ix, int iy) { return uy2_[offset(ix,iy)]; }
    float& uy3(int ix, int iy) { return uy3_[offset(ix,iy)]; }
    float& fx1(int ix, int iy) { return fx1_[offset(ix,iy)]; }           // x differences of f
    float& fx2(int ix, int iy) { return fx2_[offset(ix,iy)]; }
    float& fx3(int ix, int iy) { return fx3_[offset(ix,iy)]; }
    float& gy1(int ix, int iy) { return gy1_[offset(ix,iy)]; }           // y differences of g
    float& gy2(int ix, int iy) { return gy2_[offset(ix,iy)]; }
    float& gy3(int ix, int iy) { return gy3_[offset(ix,iy)]; }
    float& v1(int ix, int iy) {return v1_[offset(ix,iy)]; }            // Solution values at next step
    float& v2(int ix, int iy) {return v2_[offset(ix,iy)]; }
    float& v3(int ix, int iy) {return v3_[offset(ix,iy)]; }

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }
    
    // Read / write elements of simulation state
    float&       operator()(int i, int j) {
        return u1_[offset(i,j)];
    }
    
    const float& operator()(int i, int j) const {
        return u1_[offset(i,j)];
    }
    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost );
    }

    float& uwrap1(int ix, int iy)  { return u1_[ioffset(ix,iy)]; }
    float& uwrap2(int ix, int iy)  { return u2_[ioffset(ix,iy)]; }
    float& uwrap3(int ix, int iy)  { return u3_[ioffset(ix,iy)]; }

    void run(float tfinal);
    // Call f(Uxy, x, y) at each cell center to set initial conditions
    inline __declspec(target (mic)) void init();
    inline __declspec(target (mic)) void apply_periodic();
    inline __declspec(target (mic)) void compute_fg_speeds(float& cx, float& cy);
    inline __declspec(target (mic)) void limited_derivs();
    inline __declspec(target (mic)) void compute_step(int io, float dt);
};


/**
 * ## Initialization
 * 
 * Before starting the simulation, we need to be able to set the
 * initial conditions.  The `init` function does exactly this by
 * running a callback function at the center of each cell in order
 * to initialize the cell $U$ value.  For the purposes of this function,
 * cell $(i,j)$ is the subdomain 
 * $[i \Delta x, (i+1) \Delta x] \times [j \Delta y, (j+1) \Delta y]$.
 */

void Central2D::init()
{
    u1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    u2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    u3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    f1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    f2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    f3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    g1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    g2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    g3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    ux1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    ux2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    ux3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    uy1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    uy2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    uy3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    fx1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    fx2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    fx3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    gy1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    gy2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    gy3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    v1_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    v2_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    v3_ = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    //default is dam break initial condition to generate the final image. 
    #pragma omp parallel for 
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix){
            float x = (ix+0.5)*dx;
            float y = (iy+0.5)*dy;
            x -= 1;
            y -= 1;
            u1(nghost+ix,nghost+iy) = 1.0 + 0.5*(x*x + y*y < 0.25+1e-5);
            u2(nghost+ix,nghost+iy) = 0;
            u3(nghost+ix,nghost+iy) = 0;
        }
}

/**
 * ## Time stepper implementation
 * 
 * ### Boundary conditions
 * 
 * In finite volume methods, boundary conditions are typically applied by
 * setting appropriate values in ghost cells.  For our framework, we will
 * apply periodic boundary conditions; that is, waves that exit one side
 * of the domain will enter from the other side.
 * 
 * We apply the conditions by assuming that the cells with coordinates
 * `nghost <= ix <= nx+nghost` and `nghost <= iy <= ny+nghost` are
 * "canonical", and setting the values for all other cells `(ix,iy)`
 * to the corresponding canonical values `(ix+p*nx,iy+q*ny)` for some
 * integers `p` and `q`.
 */

inline __declspec(target (mic)) void Central2D::apply_periodic()
{
    // Copy data between right and left boundaries
    #pragma omp parallel for
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nghost; ++ix) {
            u1(ix,          iy) = uwrap1(ix,          iy);
            u1(nx+nghost+ix,iy) = uwrap1(nx+nghost+ix,iy);
            u2(ix,          iy) = uwrap2(ix,          iy);
            u2(nx+nghost+ix,iy) = uwrap2(nx+nghost+ix,iy);
            u3(ix,          iy) = uwrap3(ix,          iy);
            u3(nx+nghost+ix,iy) = uwrap3(nx+nghost+ix,iy);

        }

    // Copy data between top and bottom boundaries
    #pragma omp parallel for
    for (int ix = 0; ix < nx_all; ++ix)
        for (int iy = 0; iy < nghost; ++iy) {
            u1(ix,          iy) = uwrap1(ix,          iy);
            u1(ix,ny+nghost+iy) = uwrap1(ix,ny+nghost+iy);
            u2(ix,          iy) = uwrap2(ix,          iy);
            u2(ix,ny+nghost+iy) = uwrap2(ix,ny+nghost+iy);
            u3(ix,          iy) = uwrap3(ix,          iy);
            u3(ix,ny+nghost+iy) = uwrap3(ix,ny+nghost+iy);
        }
}


/**
 * ### Initial flux and speed computations
 * 
 * At the start of each time step, we need the flux values at
 * cell centers (to advance the numerical method) and a bound
 * on the wave speeds in the $x$ and $y$ directions (so that
 * we can choose a time step that respects the specified upper
 * bound on the CFL number).
 */

inline __declspec(target (mic)) void Central2D::compute_fg_speeds(float& cx_, float& cy_)
{
    float grav = 9.8;
    using namespace std;
    float cx = 1.0e-15;
    float cy = 1.0e-15;
    #pragma omp parallel for
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            float cell_cx, cell_cy;

            //calculate flux
            float h = u1(ix,iy), hu = u2(ix,iy), hv = u3(ix,iy);
            f1(ix,iy) = hu;
            f2(ix,iy) = hu*hu/h + grav *(0.5)*h*h;
            f3(ix,iy) = hu*hv/h;

            g1(ix,iy) = hv;
            g2(ix,iy) = hu*hv/h;
            g3(ix,iy) = hv*hv/h + grav *(0.5)*h*h;

            float root_gh = sqrt(grav * h);  // NB: Don't let h go negative!
            cell_cx = abs(hu/h) + root_gh;
            cell_cy = abs(hv/h) + root_gh;
            cx = max(cx, cell_cx);
            cy = max(cy, cell_cy);
        }
    cx_ = cx;
    cy_ = cy;
}

/**
 * ### Derivatives with limiters
 * 
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 */

inline __declspec(target (mic)) void Central2D::limited_derivs()
{
    #pragma omp parallel for 
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {

            // x derivs
            ux1(ix,iy) = limdiff( u1(ix-1,iy), u1(ix,iy), u1(ix+1,iy) );
            fx1(ix,iy) = limdiff( f1(ix-1,iy), f1(ix,iy), f1(ix+1,iy) );
            ux2(ix,iy) = limdiff( u2(ix-1,iy), u2(ix,iy), u2(ix+1,iy) );
            fx2(ix,iy) = limdiff( f2(ix-1,iy), f2(ix,iy), f2(ix+1,iy) );
            ux3(ix,iy) = limdiff( u3(ix-1,iy), u3(ix,iy), u3(ix+1,iy) );
            fx3(ix,iy) = limdiff( f3(ix-1,iy), f3(ix,iy), f3(ix+1,iy) );

            // y derivs
            uy1(ix,iy) = limdiff( u1(ix,iy-1), u1(ix,iy), u1(ix,iy+1) );
            gy1(ix,iy) = limdiff( g1(ix,iy-1), g1(ix,iy), g1(ix,iy+1) );
            uy2(ix,iy) = limdiff( u2(ix,iy-1), u2(ix,iy), u2(ix,iy+1) );
            gy2(ix,iy) = limdiff( g2(ix,iy-1), g2(ix,iy), g2(ix,iy+1) );
            uy3(ix,iy) = limdiff( u3(ix,iy-1), u3(ix,iy), u3(ix,iy+1) );
            gy3(ix,iy) = limdiff( g3(ix,iy-1), g3(ix,iy), g3(ix,iy+1) );
        }
}

/**
 * ### Advancing a time step
 * 
 * Take one step of the numerical scheme.  This consists of two pieces:
 * a first-order corrector computed at a half time step, which is used
 * to obtain new $F$ and $G$ values; and a corrector step that computes
 * the solution at the full step.  For full details, we refer to the
 * [Jiang and Tadmor paper][jt].
 * 
 * The `compute_step` function takes two arguments: the `io` flag
 * which is the time step modulo 2 (0 if even, 1 if odd); and the `dt`
 * flag, which actually determines the time step length.  We need
 * to know the even-vs-odd distinction because the Jiang-Tadmor
 * scheme alternates between a primary grid (on even steps) and a
 * staggered grid (on odd steps).  This means that the data at $(i,j)$
 * in an even step and the data at $(i,j)$ in an odd step represent
 * values at different locations in space, offset by half a space step
 * in each direction.  Every other step, we shift things back by one
 * mesh cell in each direction, essentially resetting to the primary
 * indexing scheme.
 */

inline __declspec(target (mic)) void Central2D::compute_step(int io, float dt)
{
    #pragma omp parallel
    {
    float dtcdx2 = 0.5 * dt / dx;
    float dtcdy2 = 0.5 * dt / dy;
    float grav =9.8;
    // Predictor (flux values of f and g at half step)
    #pragma omp for
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            float h = u1(ix,iy); float hu = u2(ix,iy); float hv = u3(ix,iy);
            h -= dtcdx2 * fx1(ix,iy);
            h -= dtcdy2 * gy1(ix,iy);
            hu -= dtcdx2 * fx2(ix,iy);
            hu -= dtcdy2 * gy2(ix,iy);
            hv -= dtcdx2 * fx3(ix,iy);
            hv -= dtcdy2 * gy3(ix,iy);
            f1(ix,iy) = hu;
            f2(ix,iy) = hu*hu/h + grav *(0.5)*h*h;
            f3(ix,iy) = hu*hv/h;

            g1(ix,iy) = hv;
            g2(ix,iy) = hu*hv/h;
            g3(ix,iy) = hv*hv/h + grav *(0.5)*h*h;
        }

    // Corrector (finish the step)
    #pragma omp for
    for (int iy = nghost-io; iy < ny+nghost-io; ++iy)
        for (int ix = nghost-io; ix < nx+nghost-io; ++ix) {
            v1(ix,iy) =
                0.2500 * ( u1(ix,  iy) + u1(ix+1,iy  ) +
                           u1(ix,iy+1) + u1(ix+1,iy+1) ) -
                0.0625 * ( ux1(ix+1,iy  ) - ux1(ix,iy  ) +
                           ux1(ix+1,iy+1) - ux1(ix,iy+1) +
                           uy1(ix,  iy+1) - uy1(ix,  iy) +
                           uy1(ix+1,iy+1) - uy1(ix+1,iy) ) -
                dtcdx2 * ( f1(ix+1,iy  ) - f1(ix,iy  ) +
                           f1(ix+1,iy+1) - f1(ix,iy+1) ) -
                dtcdy2 * ( g1(ix,  iy+1) - g1(ix,  iy) +
                           g1(ix+1,iy+1) - g1(ix+1,iy) );
        
            v2(ix,iy) =
                0.2500 * ( u2(ix,  iy) + u2(ix+1,iy  ) +
                           u2(ix,iy+1) + u2(ix+1,iy+1) ) -
                0.0625 * ( ux2(ix+1,iy  ) - ux2(ix,iy  ) +
                           ux2(ix+1,iy+1) - ux2(ix,iy+1) +
                           uy2(ix,  iy+1) - uy2(ix,  iy) +
                           uy2(ix+1,iy+1) - uy2(ix+1,iy) ) -
                dtcdx2 * ( f2(ix+1,iy  ) - f2(ix,iy  ) +
                           f2(ix+1,iy+1) - f2(ix,iy+1) ) -
                dtcdy2 * ( g2(ix,  iy+1) - g2(ix,  iy) +
                           g2(ix+1,iy+1) - g2(ix+1,iy) );

            v3(ix,iy) =
                0.2500 * ( u3(ix,  iy) + u3(ix+1,iy  ) +
                           u3(ix,iy+1) + u3(ix+1,iy+1) ) -
                0.0625 * ( ux3(ix+1,iy  ) - ux3(ix,iy  ) +
                           ux3(ix+1,iy+1) - ux3(ix,iy+1) +
                           uy3(ix,  iy+1) - uy3(ix,  iy) +
                           uy3(ix+1,iy+1) - uy3(ix+1,iy) ) -
                dtcdx2 * ( f3(ix+1,iy  ) - f3(ix,iy  ) +
                           f3(ix+1,iy+1) - f3(ix,iy+1) ) -
                dtcdy2 * ( g3(ix,  iy+1) - g3(ix,  iy) +
                           g3(ix+1,iy+1) - g3(ix+1,iy) );
        }
    // Copy from v storage back to main grid
    #pragma omp for
    for (int j = nghost; j < ny+nghost; ++j){
        for (int i = nghost; i < nx+nghost; ++i){
            u1(i,j) = v1(i-io,j-io);
            u2(i,j) = v2(i-io,j-io);
            u3(i,j) = v3(i-io,j-io);
        }
    }

    ///ending pragma parallel portion
    }
}


/**
 * ### Advance time
 * 
 * The `run` method advances from time 0 (initial conditions) to time
 * `tfinal`.  Note that `run` can be called repeatedly; for example,
 * we might want to advance for a period of time, write out a picture,
 * advance more, and write another picture.  In this sense, `tfinal`
 * should be interpreted as an offset from the time represented by
 * the simulator at the start of the call, rather than as an absolute time.
 * 
 * We always take an even number of steps so that the solution
 * at the end lives on the main grid instead of the staggered grid. 
 */



void Central2D::run(float tfinal)
{
    float *foo = (float*)malloc(NX_ALL*NX_ALL*sizeof(float));
    float sum = 0; 
    //#pragma offload target(mic) in(u1_, u2_ : length(NX_ALL*NX_ALL)) out(sum)
    #ifdef __MIC__
    #pragma offload target(mic) inout(u1_,u2_,u3_,f1_,f2_,f3_,g1_,g2_,g3_,ux1_,ux2_,ux3_,uy1_,uy2_,uy3_,fx1_,fx2_,fx3_,gy1_,gy2_,gy3_,v3_,v1_,v2_: length(NX_ALL*NX_ALL))
    {
        /*
    bool done = false;
    float t = 0;
        while (!done) {
            float dt;
            for (int io = 0; io < 2; ++io) {
                float cx, cy;
                apply_periodic();
                compute_fg_speeds(cx, cy);
                limited_derivs();
                if (io == 0) {
                    dt = cfl / std::max(cx/dx, cy/dy);
                    if (t + 2*dt >= tfinal) {
                        dt = (tfinal-t)/2;
                        done = true;
                    }
                }
                compute_step(io, dt);
                t += dt;
            }
        }
        */
        float *a = (float*)_mm_malloc(sizeof(float)*16, 64);
        float *b = (float*)_mm_malloc(sizeof(float)*16, 64);

        __mmask16  k = 0x5555;
            __m512 A, B;
        A = _mm512_maskz_loadu_ps(k, a);
        B = _mm512_maskz_loadu_ps(k, b);
        A = _mm512_add_ps(A, B);
            _mm512_store_ps( a, A);

            for(int i = 0; i < 16; i ++) printf("%f ", a[i]);
            printf("\n");
    }
    #else
    //foo
    #endif   
       
}

/**
 * ### Diagnostics
 * 
 * The numerical method is supposed to preserve (up to rounding
 * errors) the total volume of water in the domain and the total
 * momentum.  Ideally, we should also not see negative water heights,
 * since that will cause the system of equations to blow up.  For
 * debugging convenience, we'll plan to periodically print diagnostic
 * information about these conserved quantities (and about the range
 * of water heights).
 */

void Central2D::solution_check()
{
    using namespace std;
    float h_sum = 0, hu_sum = 0, hv_sum = 0;
    float hmin = u1(nghost,nghost);
    float hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i) {
            
            float h = u1(i,j);
            h_sum += h;
            hu_sum += u2(i,j);
            hv_sum += u3(i,j);
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            assert( h > 0) ;
        }
    float cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
}

//ldoc off
#endif /* CENTRAL2D_H*/

