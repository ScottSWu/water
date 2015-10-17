#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

#ifndef NX  
#define NX 200
#endif
 
#ifndef NX_ALL
#define NX_ALL 206
#endif


template <class Physics, class Limiter>
class Central2D {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny),
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(w/nx), dy(h/ny),
        cfl(cfl) {}

    // Advance from time 0 to time tfinal
    void run(real tfinal);

    // Call f(Uxy, x, y) at each cell center to set initial conditions
    void init();

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }
    
    // Read / write elements of simulation state
    vec&       operator()(int i, int j) {
        return u_[offset(i+nghost,j+nghost)];
    }
    
    const vec& operator()(int i, int j) const {
        return u_[offset(i+nghost,j+nghost)];
    }
    
private:
    static constexpr int nghost = 3;   // Number of ghost cells

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

         // Solution values at next step

    // Array accessor functions
    
    ///Naive, hard-to-read but easy to change change from 8 vectors to 24 arrays

    real u1_[NX_ALL*NX_ALL];            // Solution values
    real u2_[NX_ALL*NX_ALL];            
    real u3_[NX_ALL*NX_ALL]; 
    real f1_[NX_ALL*NX_ALL];            // Fluxes in x
    real f2_[NX_ALL*NX_ALL];
    real f3_[NX_ALL*NX_ALL]; 
    real g1_[NX_ALL*NX_ALL];            // Fluxes in y
    real g2_[NX_ALL*NX_ALL];
    real g3_[NX_ALL*NX_ALL];
    real ux1_[NX_ALL*NX_ALL];           // x differences of u
    real ux2_[NX_ALL*NX_ALL];
    real ux3_[NX_ALL*NX_ALL];
    real uy1_[NX_ALL*NX_ALL];           // y differences of u
    real uy2_[NX_ALL*NX_ALL];
    real uy3_[NX_ALL*NX_ALL];
    real fx1_[NX_ALL*NX_ALL];           // x differences of f
    real fx2_[NX_ALL*NX_ALL];
    real fx3_[NX_ALL*NX_ALL];
    real gy1_[NX_ALL*NX_ALL];           // y differences of g
    real gy2_[NX_ALL*NX_ALL];
    real gy3_[NX_ALL*NX_ALL];
    real v1_[NX_ALL*NX_ALL];            // Solution values at next step
    real v2_[NX_ALL*NX_ALL];
    real v3_[NX_ALL*NX_ALL];

    real u1(int ix, int iy) { return u1[offset(ix,iy)]; }            // Solution values
    real u2(int ix, int iy) { return u2[offset(ix,iy)]; }            
    real u3(int ix, int iy) { return u3[offset(ix,iy)]; } 
    real f1(int ix, int iy) { return f1[offset(ix,iy)]; }            // Fluxes in x
    real f2(int ix, int iy) { return f2[offset(ix,iy)]; }
    real f3(int ix, int iy) { return f3[offset(ix,iy)]; } 
    real g1(int ix, int iy) { return g1[offset(ix,iy)]; }            // Fluxes in y
    real g2(int ix, int iy) { return g2[offset(ix,iy)]; }
    real g3(int ix, int iy) { return g3[offset(ix,iy)]; }
    real ux1(int ix, int iy) { return ux1[offset(ix,iy)]; }           // x differences of u
    real ux2(int ix, int iy) { return ux2[offset(ix,iy)]; }
    real ux3(int ix, int iy) { return ux3[offset(ix,iy)]; }
    real uy1(int ix, int iy) { return uy1[offset(ix,iy)]; }           // y differences of u
    real uy2(int ix, int iy) { return uy2[offset(ix,iy)]; }
    real uy3(int ix, int iy) { return uy3[offset(ix,iy)]; }
    real fx1(int ix, int iy) { return fx1[offset(ix,iy)]; }           // x differences of f
    real fx2(int ix, int iy) { return fx2[offset(ix,iy)]; }
    real fx3(int ix, int iy) { return fx3[offset(ix,iy)]; }
    real gy1(int ix, int iy) { return gy1[offset(ix,iy)]; }           // y differences of g
    real gy2(int ix, int iy) { return gy2[offset(ix,iy)]; }
    real gy3(int ix, int iy) { return gy3[offset(ix,iy)]; }
    real v1(int ix, int iy) {return v1[offset(ix,iy)]; }            // Solution values at next step
    real v2(int ix, int iy) {return v2[offset(ix,iy)]; }
    real v3(int ix, int iy) {return v3[offset(ix,iy)]; }

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost );
    }

    real uwrap1(int ix, int iy)  { return u1[ioffset(ix,iy)]; }
    real uwrap2(int ix, int iy)  { return u2[ioffset(ix,iy)]; }
    real uwrap3(int ix, int iy)  { return u3[ioffset(ix,iy)]; }


    // Stages of the main algorithm
    void apply_periodic();
    void compute_fg_speeds(real& cx, real& cy);
    void limited_derivs();
    void compute_step(int io, real dt);

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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::init()
{
    //default is dam break initial condition to generate the final image. 
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix){
            real x = (ix+0.5)*dx;
            real y = (iy+0.5)*dy;
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::apply_periodic()
{
    // Copy data between right and left boundaries
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    real grav = 9.8;
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;

            //calculate flux
            real h = u1(ix,iy), hu = u2(ix,iy), hv = u3(ix,iy);
            f1(ix,iy) = hu;
            f2(ix,iy) = hu*hu/h + grav *(0.5)*h*h;
            f3(ix,iy) = hu*hv/h;

            g1(ix,iy) = hv;
            g2(ix,iy) = hu*hv/h;
            g3(ix,iy) = hv*hv/h + grav *(0.5)*h*h;

            real root_gh = sqrt(grav * h);  // NB: Don't let h go negative!
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::limited_derivs()
{
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {

            // x derivs
            ux1(ix,iy) = Limiter::limdiff( u1(ix-1,iy), u1(ix,iy), u1(ix+1,iy) );
            fx1(ix,iy) = Limiter::limdiff( f1(ix-1,iy), f1(ix,iy), f1(ix+1,iy) );
            ux2(ix,iy) = Limiter::limdiff( u2(ix-1,iy), u2(ix,iy), u2(ix+1,iy) );
            fx2(ix,iy) = Limiter::limdiff( f2(ix-1,iy), f2(ix,iy), f2(ix+1,iy) );
            ux3(ix,iy) = Limiter::limdiff( u3(ix-1,iy), u3(ix,iy), u3(ix+1,iy) );
            fx3(ix,iy) = Limiter::limdiff( f3(ix-1,iy), f3(ix,iy), f3(ix+1,iy) );

            // y derivs
            uy1(ix,iy) = Limiter::limdiff( u1(ix,iy-1), u1(ix,iy), u1(ix,iy+1) );
            gy1(ix,iy) = Limiter::limdiff( g1(ix,iy-1), g1(ix,iy), g1(ix,iy+1) );
            uy2(ix,iy) = Limiter::limdiff( u2(ix,iy-1), u2(ix,iy), u2(ix,iy+1) );
            gy2(ix,iy) = Limiter::limdiff( g2(ix,iy-1), g2(ix,iy), g2(ix,iy+1) );
            uy3(ix,iy) = Limiter::limdiff( u3(ix,iy-1), u3(ix,iy), u3(ix,iy+1) );
            gy3(ix,iy) = Limiter::limdiff( g3(ix,iy-1), g3(ix,iy), g3(ix,iy+1) );
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_step(int io, real dt)
{
    #pragma omp parallel
    {
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;
    real grav =9.8;
    // Predictor (flux values of f and g at half step)
    #pragma omp for
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            h = u1(ix,iy); hu = u2(ix,iy); hv = u3(ix,iy)
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
                dtcdx2 * ( f1(ix+1,iy  ) - f1(ix,iy  )1 +
                           f1(ix+1,iy+1) - f1(ix,iy+1)1 ) -
                dtcdy2 * ( g1(ix,  iy+1) - g1(ix,  iy)1 +
                           g1(ix+1,iy+1) - g1(ix+1,iy)1 );
        
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::run(real tfinal)
{
    bool done = false;
    real t = 0;
    while (!done) {
        real dt;
        for (int io = 0; io < 2; ++io) {
            real cx, cy;
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::solution_check()
{
    using namespace std;
    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = u1(nghost,nghost);
    real hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i) {
            
            real h = u1(i,j);
            h_sum += h;
            hu_sum += uij[1];
            hv_sum += uij[2];
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            assert( h > 0) ;
        }
    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
}

//ldoc off
#endif /* CENTRAL2D_H*/

