#include <iostream>
// # define MADNESS_HAS_LIBXC 0
// #define USE_GENTENSOR 0 // only needed if madness was configured with `-D ENABLE_GENTENSOR=1
// #include<madness/chem/nemo.h>
// #include<madness.h>
// #include<madness/chem.h>
// #include<madness/misc/info.h>
// #include<madness/mra/nonlinsol.h>
// #include<madness/world/timing_utilities.h>

#include<madchem.h>

using namespace madness;


enum Uplo {upper, lower};
static bool transform_c=false;
static bool debug=false;
static double alpha1=constants::fine_structure_constant;
static double shift=0.0;
static double epsilon=1.e-12;
static bool use_ble=false;
static const double bohr_rad=52917.7211;
static const double lo=1.e-7;


double compute_gamma(const double nuclear_charge) {
    return sqrt(1-nuclear_charge*nuclear_charge*alpha1*alpha1);
}

struct stepfunction : public FunctionFunctorInterface<double,3> {
    int axis=-1;
    stepfunction(const int axis) : axis(axis) {
        MADNESS_CHECK(axis>=0 && axis<3);
    }
    double operator()(const coord_3d& r) const override { return r[axis]/(r.normf()+epsilon); }
    Level special_level() override {return 20;};
    std::vector<Vector<double, 3UL>> special_points() const override {
        coord_3d o={0.0,0.0,0.0};
        return {o};
    }
};

struct ncf_cusp {
    double a=1.3;
    double Z=-1;
    ncf_cusp(double a, double Z) :a(a),Z(Z) {}
    double operator()(const double& r) const {
        if (a<0.0) return 1.0;
        return 1.0+(1.0/(a-1.0))*exp(-a*Z*r);
    }
};

struct Sigma_ncf_cusp {
    double a=1.3;
    double Z=-1;
    Sigma_ncf_cusp(double a, double Z) :a(a),Z(Z) {}
    double operator()(const double& r) const {
        if (a<0.0) return 0.0;
        const double denominator=1.0 + 1.0 / (a - 1.0) * exp(-a * Z * r);
        const double numerator=-a*Z/(a-1.0) * exp(-a * Z * r);
        return numerator/denominator;
    }
};

struct ncf_singularity {
    double gamma=0.0;
    ncf_singularity(double gamma) : gamma(gamma) {}
    double operator()(const double& r) const {
        if (gamma<0.0) return 1.0;
        return 1.0/(std::pow(r,1.0-gamma)+epsilon);
    }
};

struct ncf : public FunctionFunctorInterface<double,3> {
    ncf_singularity ncfs;
    ncf_cusp ncfc;
    long power=1;
    ncf(double gamma, double a, double Z) : ncfs(gamma), ncfc(a,Z) {}
    double operator()(const coord_3d& r) const override {
        double rr=r.normf();
        return std::pow(ncfs(rr) * ncfc(rr),power);
    }

    Level special_level() override {return 20;};
    std::vector<Vector<double, 3UL>> special_points() const override {
        coord_3d o={0.0,0.0,0.0};
        return {o};
    }
};

double generalized_laguerre(const double alpha, const long n, const double r) {
    if (n<0) return 0.0;
    else if (n==0) return 1.0;
    else if (n==1) return (1.0+alpha-r);
    else if (n==2) return 0.5*r*r - (alpha + 2.0)*r + 0.5*(alpha+1)*(alpha+2);
    else
    MADNESS_EXCEPTION("generalized Laguerre polynomial implemented only up to order n=2",1);
}

//Functor to make the fermi nuclear charge distribution (NOT the potential, and NOT normalized) for a given center
//This is based on Visscher and Dyall's 1997 paper on nuclear charge distributions.
class FermiNucDistFunctor : public FunctionFunctorInterface<double,3> {
private:
    int m_A;
    double m_T;
    double m_C;
    std::vector<coord_3d> m_R;
public:
    // Constructor
    FermiNucDistFunctor(int& Z, coord_3d R, double bohr_rad){
        //m_T = 0.000043463700858425666; //2.3 fm in bohr
        m_T = 2.3/bohr_rad;
        m_R.push_back(R);

        //find atomic mass numbers for each atom. This list matches that of Visccher and Dyall (1997)
        int Alist[116] = {1,4,7,9,11,12,14,16,19,20,23,24,27,28,31,32,35,40,39,40,45,48,51,52,55,56,59,58,63,64,69,74,75,80,79,84,85,88,89,90,93,98,98,102,103,106,107,114,115,120,121,130,127,132,133,138,139,140,141,144,145,152,153,158,159,162,162,168,169,174,175,180,181,184,187,192,193,195,197,202,205,208,209,209,210,222,223,226,227,232,231,238,237,244,243,247,247,251,252,257,258,259,262,261,262,263,262,265,266,264,272,277,284,289,288,292};
        m_A = Alist[Z-1];

        double PI = constants::pi;
        if(m_A < 5){
            m_C = 0.000022291*pow(m_A, 1.0/3.0) - 0.0000090676;
        }
        else{
            m_C = sqrt(5.0/3.0*pow((0.836*pow(m_A,1.0/3.0)+0.570)/bohr_rad,2) - 7.0/3.0*pow(PI*m_T/4.0/log(3.0),2));
        }
    }

    //overload () operator
    double operator() (const coord_3d&r) const {
        double x = r[0] - m_R[0][0];
        double y = r[1] - m_R[0][1];
        double z = r[2] - m_R[0][2];
        double rr = sqrt(x*x+y*y+z*z);
        double result = 1.0/(1.0+exp(4.0*log(3.0)*(rr-m_C)/m_T));
        return result;
    }

    //Because the distribution is only nonzero in a small window around the center, need to create a special point
    std::vector<coord_3d> special_points() const {
        return m_R;
    }

    madness::Level special_level() {
        return 18;
    }

    //Print the parameters of the Fermi nuclear charge distribution
    void print_details(World& world){

        //Constants necessary to print the details. Technically need to use bohr_rad parameter here
        int Alist[116] = {1,4,7,9,11,12,14,16,19,20,23,24,27,28,31,32,35,40,39,40,45,48,51,52,55,56,59,58,63,64,69,74,75,80,79,84,85,88,89,90,93,98,98,102,103,106,107,114,115,120,121,130,127,132,133,138,139,140,141,144,145,152,153,158,159,162,162,168,169,174,175,180,181,184,187,192,193,195,197,202,205,208,209,209,210,222,223,226,227,232,231,238,237,244,243,247,247,251,252,257,258,259,262,261,262,263,262,265,266,264,272,277,284,289,288,292};
        double T = 2.3/52917.72490083583;
        double PI = constants::pi;

        if(world.rank()==0){
            for(int i = 0; i < 116; i++){
                double RMS = (0.836*pow(Alist[i],1.0/3.0)+0.570)/52917.72490083583;
                double C;
                if(Alist[i] < 5){
                    C = 0.000022291*pow(Alist[i], 1.0/3.0) - 0.0000090676;
                }
                else{
                    C = sqrt(5.0/3.0*pow(RMS,2)-7.0/3.0*pow(PI*T/4.0/log(3.0),2));
                }
                double xi = 3.0/2.0/pow(RMS,2);
                printf("Z: %3i,  A: %3i,  RMS: %.10e,  C: %.10e,  xi: %.10e\n", i+1, Alist[i], RMS, C, xi);
            }
        }
    }
};


//Creates the fermi nuclear potential from the charge distribution. Also calculates the nuclear repulsion energy
real_function_3d make_fermi_potential(World& world, double& nuclear_repulsion_energy, const double nuclear_charge){
    real_convolution_3d op=CoulombOperator(world,lo,FunctionDefaults<3>::get_thresh());
    if(world.rank()==0) print("\n***Making a Fermi Potential***");

    //Get list of atom coordinates
//    std::vector<coord_3d> Rlist = Init_params.molecule.get_all_coords_vec();
    std::vector<coord_3d> Rlist = {{0.0,0.0,0.0}};
    std::vector<int> Zlist(Rlist.size());
    unsigned int num_atoms = Rlist.size();

    //variables for upcoming loop
    real_function_3d temp, potential;
    double tempnorm;

    //Go through the atoms in the molecule and construct the total charge distribution due to all nuclei
    for(unsigned int i = 0; i < num_atoms; i++){
//        Zlist[i] = Init_params.molecule.get_atomic_number(i);
        Zlist[i] = nuclear_charge;
        FermiNucDistFunctor rho(Zlist[i], Rlist[i],bohr_rad);
        temp = real_factory_3d(world).functor(rho).truncate_mode(0);
        tempnorm = temp.trace();
        temp.scale(-Zlist[i]/tempnorm);
        if(i == 0){
            potential = temp;
            //rho.print_details(world);
        }
        else{
            potential += temp;
        }
    }

    //Potential is found by application of the coulomb operator to the charge distribution
    potential = apply(op,potential);

    //Calculate the nuclear repulsion energy
    //It doesn't change iteration to iteration, so we want to calculate it once and store the result
    //We calculate it inside this function because here we already have access to the nuclear charges and coordinates
    nuclear_repulsion_energy = 0.0;
    double rr;
    for(unsigned int m = 0; m < num_atoms; m++){
        for(unsigned int n = m+1; n < num_atoms; n++){
            coord_3d dist = Rlist[m] - Rlist[n];
            rr = std::sqrt(dist[0]*dist[0]+dist[1]*dist[1]+dist[2]*dist[2]);
            nuclear_repulsion_energy += Zlist[m]*Zlist[n]/rr;
        }
    }
    return potential;
}

/// returns the complex value of a given spherical harmonic
struct SphericalHarmonics{
    long l, m;
    bool zero=false;
    SphericalHarmonics(const long l, const long m) : l(l), m(m) {
        if (l<0) this->l=-l-1;
        if (abs(this->m)>this->l) zero=true;
    }

    double_complex operator()(const coord_3d& xyz) const {
        if (zero) return {0.0,0.0};
        const double r=xyz.normf();
        const double r2=r*r;
        const double x=xyz[0];
        const double y=xyz[1];
        const double z=xyz[2];
        const double_complex i=double_complex(0.0,1.0);
        auto stepx=stepfunction(0);
        auto stepy=stepfunction(1);
        auto stepz=stepfunction(2);
        double_complex step_x_m_iy=stepx(xyz) - i* stepy(xyz);
        double_complex step_x_p_iy=stepx(xyz) + i* stepy(xyz);
        if ((l==0) and (m== 0)) return 0.5*sqrt(1.0/constants::pi);

//        if ((l==1) and (m==-1)) return 0.5*sqrt(1.5/constants::pi) * (x - i*y)/r;
//        if ((l==1) and (m== 0)) return 0.5*sqrt(3.0/constants::pi) * z/r;
//        if ((l==1) and (m== 1)) return -0.5*sqrt(1.5/constants::pi) * (x + i*y)/r;
        if ((l==1) and (m==-1)) return 0.5*sqrt(1.5/constants::pi) * (stepx(xyz) - i *stepy(xyz));
        if ((l==1) and (m== 0)) return 0.5*sqrt(3.0/constants::pi) * stepz(xyz);
        if ((l==1) and (m== 1)) return -0.5*sqrt(1.5/constants::pi) * (stepx(xyz) + i *stepy(xyz));

        if ((l==2) and (m==-2)) return  0.25*sqrt(7.5/constants::pi) * step_x_m_iy * step_x_m_iy;
        if ((l==2) and (m==-1)) return  0.5 *sqrt(7.5/constants::pi) * step_x_m_iy * stepz(xyz);
        if ((l==2) and (m== 0)) return  0.25*sqrt(5.0/constants::pi) * (3.0*stepz(xyz)*stepz(xyz)- 1.0);
        if ((l==2) and (m== 1)) return -0.5 *sqrt(7.5/constants::pi) * step_x_p_iy * stepz(xyz);
        if ((l==2) and (m== 2)) return  0.25*sqrt(7.5/constants::pi) * step_x_p_iy * step_x_p_iy;

        if ((l==3) and (m==-3)) return  0.125*sqrt(35/constants::pi) * step_x_m_iy * step_x_m_iy * step_x_m_iy;
        if ((l==3) and (m==-2)) return  0.25*sqrt(105/constants::pi) * step_x_m_iy * step_x_m_iy * stepz(xyz);
        if ((l==3) and (m==-1)) return  0.125*sqrt(21/constants::pi) * step_x_m_iy * (5*stepz(xyz)*stepz(xyz) - 1.0);
        if ((l==3) and (m== 0)) return  0.25*sqrt(7.0/constants::pi) * (5*stepz(xyz)*stepz(xyz)*stepz(xyz) - 3.0*stepz(xyz));
        if ((l==3) and (m== 1)) return -0.125*sqrt(21/constants::pi) * step_x_p_iy * (5*stepz(xyz)*stepz(xyz) - 1.0);
        if ((l==3) and (m== 2)) return  0.25*sqrt(105/constants::pi) * step_x_p_iy * step_x_p_iy * stepz(xyz);
        if ((l==3) and (m== 3)) return -0.125*sqrt(35/constants::pi) * step_x_p_iy * step_x_p_iy * step_x_p_iy;
        MADNESS_EXCEPTION("out of range in SphericalHarmonics",1);

        return double_complex(0.0,0.0);
    }

};

struct sgl_guess {
    long n,l,m;
    double Z;
    sgl_guess(const long n, const long l, const long m, const double Z) : n(n), l(l), m(m), Z(Z) {}

    double_complex operator()(const coord_3d& coord) const {
        double_complex Y=SphericalHarmonics(l,m)(coord);
        double r=coord.normf();
        double rho=2.0*Z*r/n;
        double R= generalized_laguerre(2.*l+1.,n-l-1,rho);
        double e=exp(-0.5*rho);
        return e*R*std::pow(rho,double(l))*Y;
    }

    double energy() const {
        return 0.5*Z*Z/(n*n);
    }

    complex_function_3d get_wf(World& world) const {
        std::vector<coord_3d> special_points(1,coord_3d({0.0,0.0,0.0}));
        complex_function_3d wf = complex_factory_3d(world).functor(*this).special_points(special_points);
        double norm=wf.norm2();
        wf.scale(1.0/norm);
        return wf;
    }



};


/// following Dyall: p 104f
struct Xi {
    long k, component, a;
    double m, l;
    Xi(const long k, const double m, const double component, const double j, const long l)
                : k(k), m(m), component(component), l(l) {
        MADNESS_CHECK(component==0 or component==1);
        a=compute_a(j,l);
    }

    static long compute_a(const double j, const long l) {
        if (j>l) return 1;
        if (j<l) return -1;
        throw;
    }

    double_complex operator()(const coord_3d& xyz) const {
        double_complex result;
        double denominator2=2.0*l+1;
        if (component==0) {
            double numerator=a*sqrt((l+0.5+a*m)/denominator2);
            auto Y=SphericalHarmonics(k,lround(m-0.5));
            result=numerator*Y(xyz);
        } else if (component==1) {
            double numerator=sqrt((l+0.5-a*m)/denominator2);
            auto Y=SphericalHarmonics(k,lround(m+0.5));
            result=numerator*Y(xyz);
        }
        return result;
    }
};

struct Omega {
    long k, component;
    double m;
    Omega(const long k, const double m, const double component) : k(k), m(m), component(component) {
        MADNESS_CHECK(component==0 or component==1);
    }
    double_complex operator()(const coord_3d& xyz) const {
        double sgnk= (k>0) ? 1.0 : -1.0;
        double_complex result;
        if (component==0) {
            double nn=sqrt((k+0.5-m)/(2*k+1));
            auto Y=SphericalHarmonics(k,lround(m-0.5));
            result=nn*Y(xyz);
        } else if (component==1) {
            double nn=sqrt((k+0.5+m)/(2*k+1));
            auto Y=SphericalHarmonics(k,lround(m+0.5));
            result=-sgnk*nn*Y(xyz);
        }
        return result;
    }
};


void set_version(const int v, const double nuclear_charge) {
    if (v==1) {
        transform_c=false;
        shift=0.0;
    } else if (v==2) {
        transform_c=false;
        shift=0.0;
    } else if (v==3) {
        transform_c=true;
        shift=-compute_gamma(nuclear_charge)/(alpha1*alpha1);
    } else {
        MADNESS_EXCEPTION("unknown version",1);
    }
}

double compute_electronic_energy(const double energy) {
    double c=1.0/alpha1;
    return energy-c*c-shift;
}


template<typename T, std::size_t NDIM>
class MyDerivativeOperator : public SCFOperatorBase<T,NDIM> {
    typedef Function<T,NDIM> functionT;
    typedef std::vector<functionT> vecfuncT;
    typedef Tensor<T> tensorT;

public:

    MyDerivativeOperator(World& world, const int axis1) : world(world), axis(axis1) {}
    void set_ble1() {ble=true;};

    std::string info() const {return "D"+std::to_string(axis);}

    functionT operator()(const functionT& ket) const {
        vecfuncT vket(1,ket);
        return this->operator()(vket)[0];
    }

    vecfuncT operator()(const vecfuncT& vket) const {
        auto gradop = free_space_derivative<T,NDIM>(world, axis);
        if (ble) gradop.set_ble1();
        vecfuncT dvket=apply(world, gradop, vket, false);
        world.gop.fence();
        return dvket;
    }

    T operator()(const functionT& bra, const functionT& ket) const {
        vecfuncT vbra(1,bra), vket(1,ket);
        Tensor<T> tmat=this->operator()(vbra,vket);
        return tmat(0l,0l);
    }

    tensorT operator()(const vecfuncT& vbra, const vecfuncT& vket) const {
        const auto bra_equiv_ket = &vbra == &vket;
        vecfuncT dvket=this->operator()(vket);
        return matrix_inner(world,vbra,dvket, bra_equiv_ket);
    }

private:
    World& world;
    int axis;
    bool ble=false;
};

template<typename T, std::size_t NDIM>
class n_times_n_dot_p_minus_p : public SCFOperatorBase<T,NDIM> {
    typedef Function<T,NDIM> functionT;
    typedef std::vector<functionT> vecfuncT;
    typedef Tensor<T> tensorT;

public:

    n_times_n_dot_p_minus_p(World& world, const int axis1) : world(world), axis(axis1) {}

    std::string info() const override {return "n"+std::to_string(axis)+" * n.p";}

    functionT operator()(const functionT& ket) const override {
        vecfuncT vket(1,ket);
        return this->operator()(vket)[0];
    }

    vecfuncT operator()(const vecfuncT& vket) const override {
        auto D0 = free_space_derivative<T,NDIM>(world,0);
        auto D1 = free_space_derivative<T,NDIM>(world,1);
        auto D2 = free_space_derivative<T,NDIM>(world,2);
        auto Daxis = free_space_derivative<T,NDIM>(world,axis);
        if (use_ble) {
            D0.set_ble1();
            D1.set_ble1();
            D2.set_ble1();
        }
        world.gop.fence();
        const long axis1=axis;
        auto stepx=stepfunction(0);
        auto stepy=stepfunction(1);
        auto stepz=stepfunction(2);
        auto step_axis=stepfunction(axis1);
        real_function_3d n0=real_factory_3d(world).functor(stepx);
        real_function_3d n1=real_factory_3d(world).functor(stepy);
        real_function_3d n2=real_factory_3d(world).functor(stepz);
        real_function_3d naxis=real_factory_3d(world).functor(step_axis);

        auto result=naxis * (n0 * apply(world,D0,vket) + n1 * apply(world,D1,vket) + n2*apply(world,D2,vket)) - apply(world,Daxis,vket);
        return truncate(result);
    }

    T operator()(const functionT& bra, const functionT& ket) const override {
        vecfuncT vbra(1,bra), vket(1,ket);
        Tensor<T> tmat=this->operator()(vbra,vket);
        return tmat(0l,0l);
    }

    tensorT operator()(const vecfuncT& vbra, const vecfuncT& vket) const override {
        const auto bra_equiv_ket = &vbra == &vket;
        vecfuncT dvket=this->operator()(vket);
        return matrix_inner(world,vbra,dvket, bra_equiv_ket);
    }

private:
    World& world;
    int axis;
};

template<typename T, std::size_t NDIM>
class AngularMomentum_with_step : public SCFOperatorBase<T,NDIM> {
    typedef Function<T,NDIM> functionT;
    typedef std::vector<functionT> vecfuncT;
    typedef Tensor<T> tensorT;

public:

    AngularMomentum_with_step(World& world, const int axis1) : world(world), axis(axis1) {}

    std::string info() const {return "1/r L"+std::to_string(axis);}

    functionT operator()(const functionT& ket) const {
        vecfuncT vket(1,ket);
        return this->operator()(vket)[0];
    }

    vecfuncT operator()(const vecfuncT& vket) const {
        // if axis==0 -> lx = y pz - z py etc
        int index1=(axis+1)%3;
        int index2=(axis+2)%3;
        auto gradop1 = free_space_derivative<T,NDIM>(world,index1);
        auto gradop2 = free_space_derivative<T,NDIM>(world,index2);
        if (use_ble) {
            gradop1.set_ble1();
            gradop2.set_ble1();
        }
        const double_complex ii={0.0,1.0};
        vecfuncT p1=-ii*apply(world, gradop1, vket);
        vecfuncT p2=-ii*apply(world, gradop2, vket);
        world.gop.fence();
        auto step1=stepfunction(index1);
        auto step2=stepfunction(index2);
        real_function_3d r1=real_factory_3d(world).functor(step1);
        real_function_3d r2=real_factory_3d(world).functor(step2);

        auto result=r1*p2 - r2*p1;
        return result;
    }

    T operator()(const functionT& bra, const functionT& ket) const {
        vecfuncT vbra(1,bra), vket(1,ket);
        Tensor<T> tmat=this->operator()(vbra,vket);
        return tmat(0l,0l);
    }

    tensorT operator()(const vecfuncT& vbra, const vecfuncT& vket) const {
        const auto bra_equiv_ket = &vbra == &vket;
        vecfuncT dvket=this->operator()(vket);
        return matrix_inner(world,vbra,dvket, bra_equiv_ket);
    }

private:
    World& world;
    int axis;
};


/// defines a 4-spinor
class Spinor {
public:
    vector_complex_function_3d components;
    World& world() const {return components.front().world();}
    Spinor() {
        components.resize(4);
    }

    Spinor(World& world) {
        components=zero_functions_compressed<double_complex,3>(world,4);
    }

    Spinor(const vector_complex_function_3d& components) : components(components){}

    Spinor& operator+=(const Spinor& other) {
        components+=other.components;
        return *this;
    }

    Spinor& operator-=(const Spinor& other) {
        components-=other.components;
        return *this;
    }

    Spinor operator+(const Spinor& other) const {
        Spinor result;
        result.components=copy(world(),components); // deep copy
        result.components+=other.components;
        return result;
    }

    Spinor operator-(const Spinor& other) const {
        Spinor result;
        result.components=copy(world(),components); // deep copy
        result.components-=other.components;
        return result;
    }

    Spinor& truncate() {
        madness::truncate(components);
        return *this;
    }

    void print_norms(std::string text) {
        auto norms = norm2s(world(), components);
        auto realnorms = norm2s(world(), real(components));
        auto imagnorms = norm2s(world(), imag(components));
//        real_function_3d x=real_factory_3d(world()).functor([](const coord_3d& r) {return r[0];});
//        real_function_3d y=real_factory_3d(world()).functor([](const coord_3d& r) {return r[1];});
//        real_function_3d z=real_factory_3d(world()).functor([](const coord_3d& r) {return r[2];});
//        auto x1=inner(this->components,x*(this->components));
//        auto y1=inner(this->components,y*(this->components));
//        auto z1=inner(this->components,z*(this->components));
//        auto x2=inner(this->components,x*x*(this->components));
//        auto y2=inner(this->components,y*y*(this->components));
//        auto z2=inner(this->components,z*z*(this->components));
        print(text,norms);
        print("  -- real,imag",realnorms,imagnorms);
//        print("  -- moments  ",x1,y1,z1,x2,y2,z2);
//        plot(text);
    }

    void plot(const std::string filename) const {
        plot_plane(world(), real(components), "Re_"+filename);
        plot_plane(world(), imag(components), "Im_"+filename);
    }

    friend double_complex inner(const Spinor& bra, const Spinor& ket) {
        return inner(bra.components,ket.components);
    }
};

template<typename T>
Spinor operator*(const T fac, const Spinor& arg) {
    return Spinor(fac*arg.components);
}

template<typename T>
Spinor operator*(const Spinor& arg, const T fac) {
    return Spinor(fac*arg.components);
}


Spinor copy(const Spinor& other) {
    return Spinor(copy(other.world(),other.components));
}

std::vector<Spinor> copy(const std::vector<Spinor>& other) {
    std::vector<Spinor> result;
    for (auto& oo : other) result.push_back(copy(oo.world(),oo.components));
    return result;
}


template<typename T>
std::vector<Spinor> operator*(const std::vector<Spinor>& arg, const T fac) {
    std::vector<Spinor> result=copy(arg);
    for (auto& r : result) r=r*fac;
    return result;
}
double_complex inner(const std::vector<Spinor>& bra, const std::vector<Spinor> ket) {
    double_complex result=0.0;
    for (int i=0; i<bra.size(); ++i) result+=inner(bra[i],ket[i]);
    return result;
}

Tensor<double_complex> matrix_inner(const std::vector<Spinor>& bra, const std::vector<Spinor> ket) {
    Tensor<double_complex> result(bra.size(),ket.size());
    for (int i=0; i<bra.size(); ++i) {
        for (int j=0; j<ket.size(); ++j) {
            result(i,j)=inner(bra[i],ket[j]);
        }
    }
    return result;
}

std::vector<Spinor> operator-=(std::vector<Spinor>& left, const std::vector<Spinor>& right) {
    for (int i=0; i<right.size(); ++i) left[i]-=right[i];
    return left;
}

std::vector<Spinor> operator+=(std::vector<Spinor>& left, const std::vector<Spinor>& right) {
    for (int i=0; i<right.size(); ++i) left[i]+=right[i];
    return left;
}

std::vector<Spinor> operator-(const std::vector<Spinor>& left, const std::vector<Spinor>& right) {
    std::vector<Spinor> result=copy(left);
    result-=right;
    return result;
}


std::vector<Spinor> truncate(std::vector<Spinor> arg) {
    for (auto& a : arg) a.truncate();
    return arg;
}

struct LProjector {
    long lmax=3;
    World& world;
    std::map<std::pair<long,long>,complex_function_3d> Ylm;
    LProjector(World& world) : world(world) {
        for (long l=0; l<lmax; ++l) {
            for (long m=-l; m<l+1; ++m) {
                SphericalHarmonics Y(l,m);
                auto lm=std::make_pair(l,m);
                auto functor=[&Y](const coord_3d& r){return Y(r)*100.0*exp(-10.0*r.normf());};
                Ylm[lm]=complex_factory_3d(world).functor(functor);
            }
        }
    }

    void analyze(const Spinor& f, const std::string text="") const {
        madness::print(text);
        for (int c=0; c<4; ++c) {
            auto bla=f.components[c];
            LProjector lproj(world);
            lproj(bla,"component "+std::to_string(c));
        }
    }

    void operator()(const complex_function_3d& f, const std::string text="") const {
        madness::print("  "+text);
        for (long l=0; l<lmax; ++l) {
            for (long m = -l; m < l+1; ++m) {
                auto lm=std::make_pair(l,m);
                const complex_function_3d& Y=Ylm.find(lm)->second;
                double_complex ovlp=inner(Y,f);
                if (std::abs(ovlp)>1.e-7) madness::print("< lm | f> ",l,m,ovlp);
            }
        }
    }
};



// The default constructor for functions does not initialize
// them to any value, but the solver needs functions initialized
// to zero for which we also need the world object.
struct spinorallocator {
    World& world;
    const int n;

    /// @param[in]	world	the world
    /// @param[in]	nn		the number of functions in a given vector
    spinorallocator(World& world, const int n) : world(world), n(n) {
    }

    /// allocate a vector of n empty functions
   std::vector<Spinor> operator()() {
        std::vector<Spinor> r(n);
        for (auto & rr : r) rr=Spinor(world);
        return r;
    }
};

/// class defining an operator in matrix form, fixed to size (4,4)
class MatrixOperator {
public:
    typedef std::vector<std::pair<double_complex,std::shared_ptr<SCFOperatorBase<double_complex,3>>>> opT;

    explicit MatrixOperator(const int i=4, const int j=4) {
        elements.resize(i);
        for (auto& e : elements) e.resize(j);
    }

    std::size_t nrow() const {return elements.size();}
    std::size_t ncol() const {
        if (elements.size()) return elements[0].size();
        return 0;
    }

    virtual Spinor operator()(const Spinor& arg) const {
        World& world=arg.components[0].world();
        double_complex norm1=inner(arg,arg);
        Spinor result;
        for (auto& c : result.components) c=complex_factory_3d(world).compressed();
        for (int i=0; i<ncol(); ++i) {
            for (int j=0; j<nrow(); ++j) {
                const opT& ops=elements[i][j];
                for (const auto& op : ops) {
                    auto fac=op.first;
                    result.components[i]+=fac * (*op.second)(arg.components[j]);
                }
            }
        }
        double_complex norm2=inner(arg,arg);
        result.truncate();

        if (std::abs(norm1-norm2)/std::abs(norm1)>1.e-10) throw;

        return result;
    }

    virtual std::vector<Spinor> operator()(const std::vector<Spinor>& arg) const {
        std::vector<Spinor> result;
        for (auto& a : arg) result.push_back((*this)(a));
        return result;
    }


    /// add a submatrix to this

    /// @param[in]  istart   row where to add the submatrix
    /// @param[in]  jstart   column where to add the submatrix
    void add_submatrix(int istart, int jstart, const MatrixOperator& submatrix) {
        if (istart+submatrix.ncol()>ncol()) throw std::runtime_error("submatrix too large: too many columns");
        if (jstart+submatrix.nrow()>nrow()) throw std::runtime_error("submatrix too large: too many rows");
        for (int i=istart; i<istart+submatrix.ncol(); ++i) {
            for (int j=jstart; j<jstart+submatrix.nrow(); ++j) {
                for (auto& elem : submatrix.elements[i-istart][j-jstart]) elements[i][j].push_back(elem);
            }
        }
    }

    MatrixOperator& operator+=(const MatrixOperator& other) {
        for (int i=0; i<ncol(); ++i) {
            for (int j=0; j<nrow(); ++j) {
                for (auto& elem : other.elements[i][j]) elements[i][j].push_back(elem);
            }
        }
        return *this;
    }

    MatrixOperator operator+(const MatrixOperator& other) const {
        MatrixOperator result;
        for (int i=0; i<ncol(); ++i) {
            for (int j=0; j<nrow(); ++j) {
                for (auto& elem : this->elements[i][j]) result.elements[i][j].push_back(elem);
                for (auto& elem : other.elements[i][j]) result.elements[i][j].push_back(elem);
            }
        }
        return result;
    }

    void add_operator(const int i, const int j, const double_complex& fac, const std::shared_ptr<SCFOperatorBase<double_complex,3>> op) {
        elements[i][j].push_back(std::make_pair(fac,op));
    }

    void print(std::string name="") const {
        madness::print(name);
        for (int i=0; i<ncol(); i++) {
            for (int j=0; j<nrow(); j++) {
                const opT& ops=elements[i][j];
                for (auto op : ops) {
                    auto fac=op.first;
                    std::cout << " " << fac << " " << (op.second)->info();
                }
                std::cout << " ||| ";
            }
            std::cout << std::endl;
        }
    }

    /// matrix containing prefactor and operator
    std::vector<std::vector<opT>> elements;
};

class Metric : public MatrixOperator{
public:
    double c0=1.0, c1=1.0, c2=1.0, c3=1.0;
    virtual Spinor operator()(const Spinor& arg) const {
        auto result=copy(arg);
        result.components[0].scale(c0);
        result.components[1].scale(c1);
        result.components[2].scale(c2);
        result.components[3].scale(c3);
        return result;
    }

    void print() const {
        madness::print("metric ",c0,c1,c2,c3);
    }
};

Metric N_metric() {
    Metric result;
    result.c0=alpha1*alpha1;
    result.c1=alpha1*alpha1;
    return result;
}
Metric M_metric() {
    Metric result;
    result.c2=alpha1*alpha1;
    result.c3=alpha1*alpha1;
    return result;
}

void show_norms(const Spinor& bra, const Spinor& ket, const std::string& name) {
    Metric m;
    if (transform_c) m=M_metric();
    auto norms = inner(ket.world(), m(bra).components, ket.components);
    madness::print("norms of ",name,": ",norms);
}

MatrixOperator make_Hdiag(World& world, const LocalPotentialOperator<double_complex,3>& V1) {
    MatrixOperator Hv;
    Hv.add_operator(0,0, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hv.add_operator(1,1, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hv.add_operator(2,2, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hv.add_operator(3,3, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    return Hv;
}
/// this is the nuclear potential on the diagonal
MatrixOperator make_Hv(World& world, const double nuclear_charge) {
    complex_function_3d V=complex_factory_3d(world)
            .functor([&nuclear_charge](const coord_3d& r){return double_complex(-nuclear_charge/(r.normf()+1.e-13));});
    auto V1=LocalPotentialOperator<double_complex,3>(world,"V",V);
    return make_Hdiag(world,V1);
}

MatrixOperator make_sigma_sn(World& world, const Uplo& uplo,
                             const LocalPotentialOperator<double_complex, 3>& x,
                             const LocalPotentialOperator<double_complex, 3>& y,
                             const LocalPotentialOperator<double_complex, 3>& z,
                             const double factor) {

    MatrixOperator H(4,4);
    const double_complex ii=double_complex(0.0,1.0);
    const double_complex one=double_complex(1.0,0.0);

    if (uplo==upper) {
        H.add_operator(0, 2, one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));
        H.add_operator(0, 3, one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
        H.add_operator(0, 3, -ii * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));

        H.add_operator(1, 2, one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
        H.add_operator(1, 2,  ii * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));
        H.add_operator(1, 3,-one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));

    } else if (uplo==lower) {
        H.add_operator(2, 0, one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));
        H.add_operator(2, 1, one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
        H.add_operator(2, 1, -ii * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));

        H.add_operator(3, 0, one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
        H.add_operator(3, 0,  ii * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));
        H.add_operator(3, 1,-one * factor, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));
    }
    return H;
}

/// returns a (4,4) hermition matrix with H=\vec \alpha \vec {xyz}
MatrixOperator make_alpha_sn(World& world,
                             const LocalPotentialOperator<double_complex, 3>& x,
                             const LocalPotentialOperator<double_complex, 3>& y,
                             const LocalPotentialOperator<double_complex, 3>& z,
                             const bool ll_negative) {
    double fac=ll_negative ? -1.0 : 1.0;

    MatrixOperator H(4,4);
    H+=make_sigma_sn(world,upper,x,y,z,1.0);
    H+=make_sigma_sn(world,lower,x,y,z,fac);

    return H;
}

/// returns a (2,2) matrix: i(gamma_1 - 1) c 1/r \vec\sigma\vec n \vec \sigma l
MatrixOperator make_snsl(World& world,const double Z) {
    MatrixOperator sp(2,2);
    const double_complex ii=double_complex(0.0,1.0);
    const double gamma1=compute_gamma(Z);
    double_complex prefac=(gamma1-1.0)*ii/alpha1;
    auto ox=n_times_n_dot_p_minus_p<double_complex,3>(world,0);
    auto oy=n_times_n_dot_p_minus_p<double_complex,3>(world,1);
    auto oz=n_times_n_dot_p_minus_p<double_complex,3>(world,2);

    sp.add_operator(0,0,    prefac,std::make_shared<n_times_n_dot_p_minus_p<double_complex,3>>(oz));
    sp.add_operator(0,1,    prefac,std::make_shared<n_times_n_dot_p_minus_p<double_complex,3>>(ox));
    sp.add_operator(0,1,-ii*prefac,std::make_shared<n_times_n_dot_p_minus_p<double_complex,3>>(oy));

    sp.add_operator(1,0,    prefac,std::make_shared<n_times_n_dot_p_minus_p<double_complex,3>>(ox));
    sp.add_operator(1,0, ii*prefac,std::make_shared<n_times_n_dot_p_minus_p<double_complex,3>>(oy));
    sp.add_operator(1,1,   -prefac,std::make_shared<n_times_n_dot_p_minus_p<double_complex,3>>(oz));
    return sp;
}



/// returns a (2,2) matrix: Z/r \vec\sigma\vec l
MatrixOperator make_Zrsl(World& world, const double Z) {
    MatrixOperator sp(2,2);
    const double_complex ii=double_complex(0.0,1.0);
    auto lx=AngularMomentum_with_step<double_complex,3>(world,0);
    auto ly=AngularMomentum_with_step<double_complex,3>(world,1);
    auto lz=AngularMomentum_with_step<double_complex,3>(world,2);

    sp.add_operator(0,0,    Z,std::make_shared<AngularMomentum_with_step<double_complex,3>>(lz));
    sp.add_operator(0,1,    Z,std::make_shared<AngularMomentum_with_step<double_complex,3>>(lx));
    sp.add_operator(0,1,-ii*Z,std::make_shared<AngularMomentum_with_step<double_complex,3>>(ly));

    sp.add_operator(1,0,    Z,std::make_shared<AngularMomentum_with_step<double_complex,3>>(lx));
    sp.add_operator(1,0, ii*Z,std::make_shared<AngularMomentum_with_step<double_complex,3>>(ly));
    sp.add_operator(1,1,   -Z,std::make_shared<AngularMomentum_with_step<double_complex,3>>(lz));
    return sp;
}

/// Hv for ansatz 1
MatrixOperator make_Hv_reg1(World& world, const double nuclear_charge) {
    const double_complex ii=double_complex(0.0,1.0);
    const double c=1.0/alpha1;
    double gamma=compute_gamma(nuclear_charge);

    complex_function_3d x_div_r2_f=complex_factory_3d(world).functor([&gamma,&c,&ii](const coord_3d& r){return -c*ii*(gamma-1)*r[0]/(inner(r,r)+epsilon);});
    complex_function_3d y_div_r2_f=complex_factory_3d(world).functor([&gamma,&c,&ii](const coord_3d& r){return -c*ii*(gamma-1)*r[1]/(inner(r,r)+epsilon);});
    complex_function_3d z_div_r2_f=complex_factory_3d(world).functor([&gamma,&c,&ii](const coord_3d& r){return -c*ii*(gamma-1)*r[2]/(inner(r,r)+epsilon);});

    auto x_div_r2=LocalPotentialOperator<double_complex,3>(world,"x/r2",x_div_r2_f);
    auto y_div_r2=LocalPotentialOperator<double_complex,3>(world,"y/r2",y_div_r2_f);
    auto z_div_r2=LocalPotentialOperator<double_complex,3>(world,"z/r2",z_div_r2_f);

    return make_alpha_sn(world,x_div_r2,y_div_r2,z_div_r2,false);

}

/// Hv for ansatz 2
MatrixOperator make_Hv_reg2(World& world, const double nuclear_charge, const double a) {
    MatrixOperator H;
    const double_complex ii=double_complex(0.0,1.0);
    const double_complex one=double_complex(1.0,0.0);
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    double gamma=compute_gamma(nuclear_charge);

    /// !!! REGULARIZATION ERROR HERE IS CRITICAL !!!
    /// !!! KEEP EPSILON SMALL !!!
    coord_3d sp{0.0, 0.0, 0.0};
    std::vector<coord_3d> special_points(1, sp);

    double Z=nuclear_charge;
    complex_function_3d x_div_r_exp=complex_factory_3d(world).functor([&a, &Z](const coord_3d& xyz){
        double r=xyz.normf();
        return xyz[0]/(r+epsilon)*(-a*Z)/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
    }).special_points(special_points).special_level(20);
    complex_function_3d y_div_r_exp=complex_factory_3d(world).functor([&a, &Z](const coord_3d& xyz){
        double r=xyz.normf();
        return xyz[1]/(r+epsilon)*(-a*Z)/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
    }).special_points(special_points).special_level(20);
    complex_function_3d z_div_r_exp=complex_factory_3d(world).functor([&a, &Z](const coord_3d& xyz){
        double r=xyz.normf();
        return xyz[2]/(r+epsilon)*(-a*Z)/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
    }).special_points(special_points).special_level(20);

    auto x_div_rexp=LocalPotentialOperator<double_complex,3>(world,"x/r slater",x_div_r_exp);
    auto y_div_rexp=LocalPotentialOperator<double_complex,3>(world,"y/r slater",y_div_r_exp);
    auto z_div_rexp=LocalPotentialOperator<double_complex,3>(world,"z/r slater",z_div_r_exp);

    H.add_operator(0,2, -ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));
    H.add_operator(0,3, -ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(0,3,-one*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));

    H.add_operator(1,2, -ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(1,2,     c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));
    H.add_operator(1,3,  ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));

    H.add_operator(2,0, -ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));
    H.add_operator(2,1, -ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(2,1,-one*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));

    H.add_operator(3,0, -ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(3,0,     c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));
    H.add_operator(3,1,  ii*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));

    return H;
}



/// the off-diagonal blocks of ansatz 3
MatrixOperator make_Hv_reg3_snZ(World& world, const double nuclear_charge, const double aa) {

    print("a in make_Hv_reg3", aa);

    // without the exponential factor we have in the (1,2) matrix:
    // i c \sn Z
    // with the exponential factor we have in the (1,2) matrix:
    // i c \sn (Z + \Sigma_U)
    // with \Sigma_U = -aZ/(a-1) exp(-aZr)/R
    // thus
    // i c \sn Z ( 1- a/(a-1) exp(-aZr)/R)

    double a=aa;
    double Z = nuclear_charge;
    double gamma= compute_gamma(Z);

    coord_3d sp{0.0, 0.0, 0.0};
    std::vector<coord_3d> special_points(1, sp);

    auto Zpot=[&Z](const double& r) {return Z;};
    Sigma_ncf_cusp ncf_cusp_pot(a,Z);

    int axis=-1;
    double factor=1.0;
    auto func =[&Zpot, &ncf_cusp_pot, &axis,&factor](const coord_3d& xyz) {
        double r = xyz.normf();
        const double_complex ii={0.0,1.0};
        const double c=1.0/alpha1;
        const double step=stepfunction(axis)(xyz);
        return ii*c * step* (Zpot(r)  +factor* ncf_cusp_pot(r));
    };

    factor=1.0;
    axis=0;
    complex_function_3d x_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);
    axis=1;
    complex_function_3d y_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);
    axis=2;
    complex_function_3d z_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);

    factor=-1.0;
    axis=0;
    complex_function_3d mx_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);
    axis=1;
    complex_function_3d my_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);
    axis=2;
    complex_function_3d mz_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);


    std::string extraname = (a > 0.0) ? " slater" : "";
    auto x_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "x/r" + extraname, x_div_r);
    auto y_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "y/r" + extraname, y_div_r);
    auto z_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "z/r" + extraname, z_div_r);

    auto mx_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "x/r" + extraname, mx_div_r);
    auto my_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "y/r" + extraname, my_div_r);
    auto mz_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "z/r" + extraname, mz_div_r);

    MatrixOperator H(4,4);
    H+=make_sigma_sn(world,upper,mx_div_rexp,my_div_rexp,mz_div_rexp,1.0);
    H+=make_sigma_sn(world,lower,x_div_rexp,y_div_rexp,z_div_rexp,-1.0);
    return H;

}

/// Hv for ansatz 3
MatrixOperator make_Hv_reg3_version1(World& world, const double nuclear_charge, const double aa) {

    MatrixOperator Hz=make_Hv_reg3_snZ(world,nuclear_charge,aa);

    MatrixOperator sl_matrix;
    auto sl= make_Zrsl(world,nuclear_charge);
    sl_matrix.add_submatrix(0, 0, sl);
    sl_matrix.add_submatrix(2, 2, sl);

    MatrixOperator snsl_matrix;
    auto snsl=make_snsl(world,nuclear_charge);
    snsl_matrix.add_submatrix(0, 2, snsl);
    snsl_matrix.add_submatrix(2, 0, snsl);


    double gamma = compute_gamma(nuclear_charge);
    double c=1.0/alpha1;

    complex_function_3d V = complex_factory_3d(world).functor(
            [&c](const coord_3d& r) { return c*c*double_complex(1.0, 0.0); });
    auto V1 = LocalPotentialOperator<double_complex, 3>(world, "(gamma-1) -- c2", V);
    auto V2 = LocalPotentialOperator<double_complex, 3>(world, "-(gamma-1) -- c2", V);
    MatrixOperator diagonal;
    diagonal.add_operator(0, 0,  (gamma - 1.0) , std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
    diagonal.add_operator(1, 1,  (gamma - 1.0) , std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
    diagonal.add_operator(2, 2, -(gamma - 1.0) , std::make_shared<LocalPotentialOperator<double_complex, 3>>(V2));
    diagonal.add_operator(3, 3, -(gamma - 1.0) , std::make_shared<LocalPotentialOperator<double_complex, 3>>(V2));
    diagonal.print("diagonal");
    Hz.print("Hz");
    sl_matrix.print("sl_matrix");
    snsl_matrix.print("snsl_matrix");
    MatrixOperator H = diagonal + Hz + sl_matrix + snsl_matrix;
    return H;
}

/// Hv for ansatz 3
MatrixOperator make_Hv_reg3_version3(World& world, const double nuclear_charge, const double aa) {
    MatrixOperator H=make_Hv_reg3_snZ(world,nuclear_charge,aa);

    double gamma = compute_gamma(nuclear_charge);

    complex_function_3d V = complex_factory_3d(world).functor(
            [](const coord_3d& r) { return double_complex(1.0, 0.0); });
    auto V1 = LocalPotentialOperator<double_complex, 3>(world, "(gamma-1)c2", V);
    auto V2 = LocalPotentialOperator<double_complex, 3>(world, "-2 gamma", V);
//    H.add_operator(0, 0, (gamma - 1.0) * c * c, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
//    H.add_operator(1, 1, (gamma - 1.0) * c * c, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
//    if (transform_c) c = 1.0;
    H.add_operator(2, 2, -2.0*gamma, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V2));
    H.add_operator(3, 3, -2.0*gamma, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V2));
    return H;

}



/// returns a (2,2) matrix
MatrixOperator make_sp(World& world) {
    MatrixOperator sp(2,2);
    const double_complex ii=double_complex(0.0,1.0);
    const double alpha=constants::fine_structure_constant;
    const double c = transform_c ? 1.0 : 1.0/alpha;
    auto Dz=MyDerivativeOperator<double_complex,3>(world,2);
    auto Dx=MyDerivativeOperator<double_complex,3>(world,0);
    auto Dy=MyDerivativeOperator<double_complex,3>(world,1);
    if (use_ble) {
        Dz.set_ble1();
        Dx.set_ble1();
        Dy.set_ble1();
    }

    sp.add_operator(0,0,-c*ii,std::make_shared<MyDerivativeOperator<double_complex,3>>(Dz));
    sp.add_operator(0,1,-c*ii,std::make_shared<MyDerivativeOperator<double_complex,3>>(Dx));
    sp.add_operator(0,1,-c,std::make_shared<MyDerivativeOperator<double_complex,3>>(Dy));

    sp.add_operator(1,0,-c*ii,std::make_shared<MyDerivativeOperator<double_complex,3>>(Dx));
    sp.add_operator(1,0,c,std::make_shared<MyDerivativeOperator<double_complex,3>>(Dy));
    sp.add_operator(1,1,c*ii,std::make_shared<MyDerivativeOperator<double_complex,3>>(Dz));
    return sp;
}


MatrixOperator make_alpha_p(World& world) {
    MatrixOperator Hd;
    MatrixOperator sp = make_sp(world);
    Hd.add_submatrix(0, 2, sp);
    Hd.add_submatrix(2, 0, sp);
    return Hd;
}

/// this is c sigma p + beta m c^2

/// @param[in]  ll  scalar on the LL (1,1) block
/// @param[in]  ss  scalar on the SS (2,2) block
MatrixOperator make_Hd(World& world, const std::pair<double_complex,std::string>& ll,
                       const std::pair<double_complex,std::string>& ss) {
    MatrixOperator Hd=make_alpha_p(world);

    complex_function_3d V=complex_factory_3d(world).functor([](const coord_3d& r){return double_complex(1.0,0.0);});
    auto V1=LocalPotentialOperator<double_complex,3>(world,ll.second,V);
    auto V2=LocalPotentialOperator<double_complex,3>(world,ss.second,V);

    if (ll.first!=0.0) {
        Hd.add_operator(0,0,ll.first,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
        Hd.add_operator(1,1,ll.first,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    }
    if (ss.first!=0.0) {
        Hd.add_operator(2, 2, ss.first, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
        Hd.add_operator(3, 3, ss.first, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
    }
    return Hd;
}

template<typename ansatzT>
std::vector<Spinor> schrodinger2dirac(const std::vector<complex_function_3d> wf, const ansatzT& ansatz, const double nuclear_charge) {
    World& world=wf.front().world();
    std::vector<Spinor> sgl;
    for (auto& w : wf ){
        Spinor tmp(world);
        tmp.components[0]=w;
        sgl.push_back(tmp);
    }
    MatrixOperator sp=make_alpha_p(world);

    std::vector<Spinor> result1;
    for (auto& s : sgl)  {
        Spinor tmp=(sp(s)+s).truncate();
        tmp.components[2].scale(0.5*alpha1*alpha1);
        tmp.components[3].scale(0.5*alpha1*alpha1);
        result1.push_back(tmp);
    }
    MatrixOperator Rinv=ansatz.Rinv(world);
    std::vector<Spinor> result=Rinv(result1);

    for (int i=0; i<result.size(); ++i) result[i].print_norms("dirac"+std::to_string(i));
    return result;
}

Tensor<double_complex> get_fock_transformation(World& world, const std::vector<Spinor>& spinors,
                                               const Tensor<double_complex>& overlap, const Tensor<double_complex>& fock) {

    const double thresh_degenerate=1.e-6;
    Tensor<double_complex> U;
    Tensor<double> evals;
    sygvp(world, fock, overlap, 1, U, evals);
    print("U before fixing");
    print(U);

    Tensor<double> occ(spinors.size());
    occ=1.0;
    Localizer::undo_reordering(U, occ, evals);
    Localizer::undo_degenerate_rotations(U, evals, thresh_degenerate);
    print("U after fixing");
    print(U);

    world.gop.broadcast(U.ptr(), U.size(), 0);
    world.gop.broadcast(evals.ptr(), evals.size(), 0);
    return U;
}

/// Transforms a vector of functions according to new[i] = sum[j] old[j]*c[j,i]

/// Uses sparsity in the transformation matrix --- set small elements to
/// zero to take advantage of this.
std::vector<Spinor>
transform(World& world, const std::vector<Spinor>& v,
          const Tensor<double_complex>& c, bool fence=true) {

    PROFILE_BLOCK(Vtransformsp);
    int n = v.size();  // n is the old dimension
    int m = c.dim(1);  // m is the new dimension
    MADNESS_ASSERT(n==c.dim(0));

    std::vector<Spinor> vc(m);
    for (int i=0; i<vc.size(); ++i) vc[i]=Spinor(world);

    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            if (c(j,i) != double_complex(0.0)) gaxpy(world,double_complex(1.0),vc[i].components,
                                                     c(j,i),v[j].components,false);
        }
    }

    if (fence) world.gop.fence();
    return vc;
}

std::vector<Spinor> orthonormalize_fock(const std::vector<Spinor>& arg,
                                        const std::vector<Spinor>& bra,
                                        Tensor<double_complex>& fock) {
    World& world=arg.front().world();
    Tensor<double_complex> ovlp(arg.size(),arg.size());
    for (int i=0; i<arg.size(); ++i) {
        for (int j=i; j<arg.size(); ++j) {
            ovlp(i,j)=inner(bra[i],arg[j]);
            ovlp(j,i)=std::conj(ovlp(i,j));
        }
    }

    Tensor<double_complex> U= get_fock_transformation(world,arg,ovlp,fock);
    fock=inner(conj(U),inner(fock,U),0,0);
    print("fock in orthonormalize_fock");
    print(fock);

    std::vector<Spinor> result=transform(world,arg,U);
    return result;

}

struct AnsatzBase {
public:
    [[nodiscard]] virtual std::string filename() const {return "ansatz"+this->name(); }
    [[nodiscard]] virtual std::string name() const =0;

    AnsatzBase(const double Z, const double a) : nuclear_charge(Z), a(a) {}
    int iansatz=0;
    double a=-1.3;
    double nuclear_charge=0.0;

    virtual void normalize(Spinor& bra, Spinor& ket) const {
        Metric m;
        if (transform_c) m=M_metric();
        double_complex norm2=inner(bra,m(ket));
        double norm=sqrt(real(norm2));
        scale(ket.world(),ket.components,1.0/norm);
        if (&bra!=&ket) scale(bra.world(),bra.components,1.0/norm);
    }

    virtual void normalize(Spinor& ket) const {
        auto bra=make_bra(ket);
        normalize(bra,ket);
    }

    virtual void normalize(std::vector<Spinor>& ket) const {
        auto bra=make_vbra(ket);
        normalize(bra,ket);
    }

    virtual void normalize(std::vector<Spinor>& bra, std::vector<Spinor>& ket) const {
        for (int i=0; i<ket.size(); ++i) normalize(bra[i],ket[i]);
    }


    virtual Spinor make_guess(World& world) const = 0;
    virtual MatrixOperator make_Hd(World& world) const = 0;
    virtual MatrixOperator R(World& world) const {
        MADNESS_EXCEPTION("no R implemented in this ansatz",1);
    }
    virtual MatrixOperator Rinv(World& world) const {
        MADNESS_EXCEPTION("no Rinv implemented in this ansatz",1);
    }

    virtual std::vector<Spinor> make_vbra(const std::vector<Spinor>& ket) const {
        std::vector<Spinor> result;
        for (const auto& k : ket) result.push_back(this->make_bra(k));
        return truncate(result);
    }
    [[nodiscard]] virtual Spinor make_bra(const Spinor& ket) const = 0;

    [[nodiscard]] virtual double mu(const double energy) const {
        return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
    }
    [[nodiscard]] double get_cusp_a() const {return a;}
};

struct Ansatz0 : public AnsatzBase {
public:

    Ansatz0(const double nuclear_charge, const double a) : AnsatzBase(nuclear_charge,a) {
        this->a=-1.3;
        iansatz=0;
    };

    [[nodiscard]] std::string name() const {
        return "0";
    }
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double_complex one(1.0,0.0);
        const double n=1;
        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma=compute_gamma(nuclear_charge);
//        print("gamma-1",gamma-1.0);
        const double C=nuclear_charge/n;
        // m=0.5;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*one*(1+gamma)*exp(-C*r.normf());});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*0.0*one;});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
        // m=-0.5;
//        print("make_guess with m=-0.5");
//        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*0.0*one;});
//        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*one*(1+gamma)*exp(-C*r.normf());});
//        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*(r[0] - ii*r[1])/r.normf()*exp(-C*r.normf());});
//        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return -std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world) const {
        return ::make_Hv(world,nuclear_charge);
    }

    Spinor make_bra(const Spinor& ket) const {
        return Spinor(copy(ket.world(),ket.components));
    }
    MatrixOperator make_Hd(World& world) const {
        double c2=1.0/(alpha1*alpha1);
        return ::make_Hd(world,{c2,"mc2"},{-c2,"-mc2"});
    }

//    double mu(const double energy) const {
//        return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
//    }
    double energy() const {
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }

    MatrixOperator R(World& world) const {
        complex_function_3d one1=complex_factory_3d(world).functor([](const coord_3d& r) {return double_complex(1.0,0.0);});
        auto one = LocalPotentialOperator<double_complex, 3>(world, "1" , one1);
        return make_Hdiag(world,one);
    }
    MatrixOperator Rinv(World& world) const {
        complex_function_3d one1=complex_factory_3d(world).functor([](const coord_3d& r) {return double_complex(1.0,0.0);});
        auto one = LocalPotentialOperator<double_complex, 3>(world, "1" , one1);
        return make_Hdiag(world,one);
    }

};

struct Ansatz1 : public AnsatzBase {
public:
    Ansatz1(const double nuclear_charge, const double a=0.0) : AnsatzBase(nuclear_charge,-1.3) {
        iansatz=1;
        if (a!=0.0) print("Ansatz 1 constructor ignores the cusp length scale a!");
    }
    std::string name() const {
        return "1";
    }
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;
        const double Z=nuclear_charge;
        const double gamma= compute_gamma(nuclear_charge);
        const double C=nuclear_charge/n;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&C,&ii](const coord_3d& r){return double_complex((1+gamma)*exp(-C*r.normf()),0.0);});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&C,&ii](const coord_3d& r){return double_complex(0.0,0.0);});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&C,&ii](const coord_3d& r){return ii*Z*alpha1*r[2]/r.normf()*exp(-C*r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&C,&ii](const coord_3d& r){return ii*Z*alpha1*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world) const {
        auto Hv=::make_Hv(world,nuclear_charge);
        Hv+= make_Hv_reg1(world,nuclear_charge);
        return Hv;
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double gamma=compute_gamma(nuclear_charge);
        ncf ncf1(gamma, a, nuclear_charge);
        real_function_3d r2=real_factory_3d(world).functor(ncf1);
        Spinor result=Spinor(r2*ket.components);
        return result;
    }
    MatrixOperator make_Hd(World& world) const {
        double c2=1.0/(alpha1*alpha1);
        return ::make_Hd(world,{c2,"mc2"},{-c2,"-mc2"});
    }

    double energy() const {
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }

    MatrixOperator Rinv(World& world) const {
        MADNESS_EXCEPTION("no Rinv in ansatz1",1);
        const double gamma=compute_gamma(nuclear_charge);

        auto ncf = [&gamma](const coord_3d& r){
            return 1.0;
        };
        complex_function_3d r1 = complex_factory_3d(world).functor(ncf);
        auto r = LocalPotentialOperator<double_complex, 3>(world, "Rinv" , r1);
        return make_Hdiag(world,r);
    }
};

struct Ansatz2 : public AnsatzBase {
public:
    Ansatz2(const double nuclear_charge, const double a) : AnsatzBase(nuclear_charge,a) {
        iansatz=2;
    }

    std::string name() const {
        return "2";
    }
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;
        const double Z=nuclear_charge;
        const double gamma= compute_gamma(nuclear_charge);
        ncf_cusp ncf(a,nuclear_charge);
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&ii,&ncf](const coord_3d& r){return double_complex((1+gamma)*exp(-Z*r.normf())/ncf(r.normf()),0.0);});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&ii,&ncf](const coord_3d& r){return double_complex(0.0,0.0)*exp(-Z*r.normf())/ncf(r.normf());});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&ii,&ncf](const coord_3d& r){return ii*Z*alpha1*r[2]/r.normf()*exp(-Z*r.normf())/ncf(r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&ii,&ncf](const coord_3d& r){return ii*Z*alpha1*(r[0] + ii*r[1])/r.normf()*exp(-Z*r.normf())/ncf(r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world) const {
        auto Hv=::make_Hv(world,nuclear_charge);
        Hv+= make_Hv_reg1(world,nuclear_charge);
        Hv+= make_Hv_reg2(world,nuclear_charge,a);
        return Hv;
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double alpha=constants::fine_structure_constant;
        const double gamma= compute_gamma(nuclear_charge);
        const double n=1;
        const double C=nuclear_charge/n;
        const double aa=a;
        coord_3d sp{0.0,0.0,0.0};
        std::vector<coord_3d> special_points(1,sp);
        real_function_3d r2=real_factory_3d(world)
                .functor([&gamma, &C, &aa](const coord_3d& r){
                    double R=std::pow(r.normf(),(gamma-1)) * (1.0+1.0/(aa-1.0)*  exp(-aa*C*r.normf()));
                    return R*R;
                })
                .special_level(15).special_points(special_points);
        Spinor result=Spinor(r2*ket.components);
        return result;
    }
    MatrixOperator make_Hd(World& world) const {
        double c2=1.0/(alpha1*alpha1);
        return ::make_Hd(world,{c2,"mc2"},{-c2,"-mc2"});
    }
//    double mu(const double energy) const {
//        return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
//    }
    double energy() const {
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }

    MatrixOperator Rinv(World& world) const {
        MADNESS_EXCEPTION("no Rinv in ansatz2",1);

        const double gamma=compute_gamma(nuclear_charge);
        const double Z=nuclear_charge;
        ncf_cusp ncf_cusp1(a,Z);
        ncf_singularity ncf_singularity1(gamma);
        complex_function_3d r1;
        auto r = LocalPotentialOperator<double_complex, 3>(world, "R" , r1);
        return make_Hdiag(world,r);
    }
};

MatrixOperator moments(World& world, int axis, int order) {
    MatrixOperator result;
    int a=axis;
    int o=order;
    complex_function_3d m=complex_factory_3d(world)
            .functor([&axis, &order](const coord_3d& r){return double_complex(std::pow(r[axis],double(order)),0.0);});
    auto mm=LocalPotentialOperator<double_complex,3>(world,"moment",m);
    result.add_operator(0,0,1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(mm));
    result.add_operator(1,1,1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(mm));
    result.add_operator(2,2,1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(mm));
    result.add_operator(3,3,1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(mm));
    return result;
}

struct Ansatz3 : public AnsatzBase {
public:
    int version=3;

    std::string name() const {
        std::string v;
        if (version==1) v=", version 1, no transform, no shift, a="+std::to_string(a);
        if (version==2) v=", version 2, no transform, partition with Hv diagonal elements zero, a="+std::to_string(a);
        if (version==3) v=", version 3, shift by gamma c^2, then ST, a="+std::to_string(a);
        return std::string("3")+v;
    }
    std::string filename() const {
        return "ansatz3";
    }

    Ansatz3(const double nuclear_charge, const double a, const int version=1)
            : AnsatzBase(nuclear_charge,a), version(version) {
        iansatz=3;
        set_version(version,nuclear_charge);
    }

    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;
        coord_3d sp{0.0,0.0,0.0};
        std::vector<coord_3d> special_points(1,sp);

        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma= compute_gamma(nuclear_charge);
        const double C=0.95*nuclear_charge/n;


        real_function_3d zero=real_factory_3d(world).functor([](const coord_3d& r){return 0.0;});

        // direct projection fails for Z>40: compute Z/2 and square afterwords, or do it twice..
        double Chalf=C;
        double N=std::pow(C,1.5)/sqrt(constants::pi);
        int counter=0;
        while (Chalf>40) {
            Chalf*=0.5;
            N=sqrt(N);
            counter++;
        }
        print("counter",counter);
        real_function_3d bla=real_factory_3d(world)
                .functor([&Chalf,&N](const coord_3d& r){return N*exp(-Chalf*r.normf());})
                .special_level(30).special_points(special_points);

        for (int i=0; i<counter; ++i) bla=bla.square();
        double nn=bla.norm2();
        print("norm of real(guess)",nn);
        result.components[0]=convert<double,double_complex>(bla);
        result.components[1]=convert<double,double_complex>(zero);
        result.components[2]=convert<double,double_complex>(0.01*bla);
        result.components[3]=convert<double,double_complex>(zero);

        double norm=norm2(world,result.components);
        print("norm of guess ",norm);
        return result;
    }


    MatrixOperator make_Hv(World& world) const {
        if (version==1)  return ::make_Hv_reg3_version1(world,nuclear_charge,a);
        if (version==2)  return ::make_Hv_reg3_snZ(world,nuclear_charge,a);
        if (version==3)  return ::make_Hv_reg3_version3(world,nuclear_charge,a);
        MADNESS_EXCEPTION("no version in ansatz 3 given",1);
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double gamma= compute_gamma(nuclear_charge);

        ncf ncf_all(gamma,a,nuclear_charge);
        ncf_all.power=2;

        real_function_3d r2=real_factory_3d(world).functor(ncf_all);
        Spinor result=2.0/(gamma+1)*Spinor(r2*ket.components);
        return result;
    }

    MatrixOperator make_Hd(World& world) const {
        double c2=1.0/(alpha1*alpha1);
        double gamma= compute_gamma(nuclear_charge);
        if (version==1) return ::make_Hd(world,{c2,"mc2"},{-c2,"-mc2"});
        if (version==2) return ::make_Hd(world,{gamma*c2,"gamma c2"},{-gamma*c2,"-gamma c2"});
        if (version==3) return ::make_Hd(world,{0.0,"mc2"},{0.0,"-mc2"});
        MADNESS_EXCEPTION("no version in ansatz 3 given",1);

    }

//    double mu(const double energy) const {
//        double gamma= compute_gamma(nuclear_charge);
//        if (version==1) return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
//        if (version==2) return sqrt(-energy*energy*alpha1*alpha1 + gamma*gamma/(alpha1*alpha1));
//        if (version==3) return sqrt(energy*energy*alpha1*alpha1);
//        MADNESS_EXCEPTION("no version in ansatz 3 given",1);
//    }
    double energy() const {
        double gamma= compute_gamma(nuclear_charge);
        if (version==1) return gamma/(alpha1*alpha1);
        if (version==2) return gamma/(alpha1*alpha1);
        if (version==3) return compute_electronic_energy(gamma/(alpha1*alpha1));
        MADNESS_EXCEPTION("no version in ansatz 3 given",1);
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }


    MatrixOperator Rinv(World& world) const {

        double gamma= compute_gamma(nuclear_charge);
        double c=1.0/alpha1;
        double_complex ii={0.0,1.0};
        double_complex prefac=0.5*(gamma+1) ;
        double_complex fac=prefac* (-1.0)*ii *c/nuclear_charge*(gamma-1);
        ncf_cusp ncf_cusp1(a,nuclear_charge);
        ncf_singularity ncf_singularity1(gamma);
        auto ncf = [&ncf_singularity1,&ncf_cusp1](const coord_3d& r) {
            return ncf_singularity1(r.normf())*ncf_cusp1(r.normf());
        };

        complex_function_3d x1=complex_factory_3d(world).functor([&fac,&ncf](const coord_3d& r){return fac*stepfunction(0)(r) / ncf(r);});
        complex_function_3d y1=complex_factory_3d(world).functor([&fac,&ncf](const coord_3d& r){return fac*stepfunction(1)(r) / ncf(r);});
        complex_function_3d z1=complex_factory_3d(world).functor([&fac,&ncf](const coord_3d& r){return fac*stepfunction(2)(r) / ncf(r);});

        auto x = LocalPotentialOperator<double_complex, 3>(world, "Rx" , x1);
        auto y = LocalPotentialOperator<double_complex, 3>(world, "Ry" , y1);
        auto z = LocalPotentialOperator<double_complex, 3>(world, "Rz" , z1);
        auto R=make_alpha_sn(world,x,y,z,false);

        complex_function_3d diag1=complex_factory_3d(world).functor([&ncf,&prefac](const coord_3d& r){return prefac/ncf(r);});
        auto diag = LocalPotentialOperator<double_complex, 3>(world, "Rdiag" , diag1);
        R+=make_Hdiag(world,diag);
        return R;
    }
};

struct ExactSpinor : public FunctionFunctorInterface<double_complex,3> {
    long n, k, l;
    mutable int component=0;
    double E=0.0, C=0.0, gamma=0.0, Z, j, m;
    bool regularized=true;
    double cusp_a=-1.0;
    bool compute_F=false;
    ExactSpinor(const int n, const char lc, const double j, const int Z, const double m=0.0)
            : ExactSpinor(n, l_char_to_int(lc),j,Z,m) { }
    ExactSpinor(const int n, const int l, const double j, const int Z,const double m=0.0)
            : n(n), l(l), j(j), Z(Z), m(m) {
        if (std::abs(j-(l+0.5))<1.e-10) k=lround(-j-0.5);       // j = l+1/2
        else k=lround(j+0.5);

        if (m==0.0) this->m=j;
        gamma=sqrt(k*k - Z*Z*alpha1*alpha1);
        E= compute_energy();
        C=compute_C();
    }

    std::string l_to_string(const long l) const {
        if (l==0) return "S";
        if (l==1) return "P";
        if (l==2) return "D";
        return "failed";
    }

    std::string filename() const {
        return "es_"+std::to_string(n)+l_to_string(l)+std::to_string(j)+"_m"+std::to_string(m);
        return "es_n"+std::to_string(n)+"_k"+std::to_string(k)+"_j"+std::to_string(j)+"_m"+std::to_string(m);
    }

    void set_ansatz(const AnsatzBase& ansatz) {
        compute_F =  (ansatz.iansatz==3) ? true  : false;
        cusp_a=ansatz.get_cusp_a();
        regularized= (ansatz.iansatz==0) ? false : true;
    }

    int l_char_to_int(const char lc) const {
        int ll=0;
        if (lc=='S') ll=0;
        else if (lc=='P') ll=1;
        else if (lc=='D') ll=2;
        else {
            MADNESS_EXCEPTION("confused L quantum in ExactSpinor",1);
        }
        return ll;
    }

    Level special_level() override {return 20;};
    std::vector<Vector<double, 3UL>> special_points() const override {
        coord_3d o={0.0,0.0,0.0};
        return {o};
    }

    double compute_energy() const {
        double c=1.0/alpha1;
        MADNESS_CHECK(gamma!=0.0);
        double E=c * c * 1.0/sqrt(1.0 + Z*Z/(c*c)*std::pow(n-std::abs(k)+gamma,-2.0));
        return E;
    }
    double compute_C() const {
        MADNESS_CHECK(E!=0.0);
        const double c=1.0/alpha1;
        return sqrt(c*c - E*E/(c*c));
    }
    // the energy dependent term for the spinor
    double compute_en() const {
        const double c=1.0/alpha1;
        return (gamma * c*c - k*get_energy()) / (c*C);
    }

    double get_energy() const {
        return E;
    }

    void print() const {
        madness::print("exact solution for n=",n,", k=",k, "j=",j, "m=",m);
        char lc='S';
        if (l==1) lc='P';
        if (l==2) lc='D';
        MADNESS_CHECK(l<3);
        madness::print("term symbol",n,lc,j);
        madness::print("energy =    ",E, E-1.0/(alpha1*alpha1));
        madness::print("compute_F   ",compute_F);
        madness::print("regularized ",regularized);
        madness::print("cusp_a      ",cusp_a);
        madness::print("C        ",C);
    }

    double_complex operator()(const coord_3d& c) const override {
        if (compute_F) return Fvalue(c);
        else return psivalue(c);
    }

    double_complex Fvalue(const coord_3d& coord) const {
        double r=coord.normf();
        double rho=2*C*r;
        double gamma1= compute_gamma(Z);
        ncf_cusp cusp(cusp_a, Z);
        double radial=exp(-rho*0.5)/cusp(r);
        if (k*k>1) radial*=std::pow(rho,gamma-gamma1);

        double_complex i={0.0,1.0};

        const long absk=std::abs(k);
        const double c=1.0/alpha1;
        const double Lnk1= generalized_laguerre(2*gamma+1.0,n-absk-1,rho);
        const double Lnk = generalized_laguerre(2*gamma-1.0,n-absk  ,rho);
        const double En=compute_en();

        const double G=Z/c * (gamma1 + gamma + 1 -k)*rho*Lnk1 + (gamma1+1)* (gamma - gamma1 -k +1)*En *Lnk;
        const double_complex F=i*(gamma1+1)*(gamma1+gamma - 1 - k) * rho * Lnk1 + i*Z/c *(gamma1- gamma + 1 + k) *En *Lnk;

        if (component==0) {
            return radial * G * Omega(k,m,0)(coord);
        } else if (component==1) {
            return radial * G * Omega(k,m,1)(coord);
        } else if (component==2) {
            return radial * F * Omega(-k,m,0)(coord);
        } else if (component==3) {
            return radial * F * Omega(-k,m,1)(coord);
        }
        MADNESS_EXCEPTION("confused component in ExactSpinor::Fvalue",1);
        return 0.0;
    }

    double_complex psivalue(const coord_3d& coord) const {
        double r=coord.normf();
        double rho=2*C*r;
        double radial=1.0;
        ncf_cusp ncf(cusp_a,Z);
        radial*=std::pow(2*C,gamma)*exp(-0.5*rho)/ncf(r);
        // three cases for R^{-1} * r^{gamma_k-1}:
        // 1. regularization with R=r^{gamma1-1}, |k|==1: factor: 1
        // 2. regularization with R=r^{gamma1-1}, |k|>1: factor: r^{gamma_k-gamma1}
        // 3. no regularization, factor r^{gamma_k-1};
        double gamma1= compute_gamma(Z);
        long absk= std::abs(k);
        if (regularized) {
            if (absk>1) {
                radial *= std::pow(r,gamma-gamma1);          // exponent is positive
            }
        } else {
            ncf_singularity ncf_s(gamma);
            radial *= ncf_s(r);
        }


        const double Lnk1= generalized_laguerre(2*gamma+1.0,n-absk-1,rho);
        const double Lnk = generalized_laguerre(2*gamma-1.0,n-absk  ,rho);
        const double En=compute_en();
        const double c=1.0/alpha1;

        double_complex i={0.0,1.0};
        double g=radial * (Z/c*rho* Lnk1 + (gamma - k)*En * Lnk);
        double f=radial * ((gamma-k)*rho*Lnk1 + Z/c*En * Lnk);
//        double f=Z*alpha1*radial;
        double sgnk= (k>0) ? 1.0 : -1.0;

//        return angular(coord,g,f);
//        if (component==0) {
//            return g * Xi(k,m,0,j,l)(coord);
//        } else if (component==1) {
//            return g * Xi(k,m,1,j,l)(coord);
//        } else if (component==2) {
//            return i * f * Xi(-k,m,0,j,l)(coord);
//        } else if (component==3) {
//            return i * f * Xi(-k,m,1,j,l)(coord);
//        }

        if (component==0) {
            return g * Omega(k,m,0)(coord);
        } else if (component==1) {
            return g * Omega(k,m,1)(coord);
        } else if (component==2) {
            return i * f * Omega(-k,m,0)(coord);
        } else if (component==3) {
            return i * f * Omega(-k,m,1)(coord);
        }

        MADNESS_EXCEPTION("confused component in ExactSpinor",1);
        return {0.0,0.0};
    }

    double_complex angular(const coord_3d& c, const double g, const double f) const {

        double_complex i={0.0,1.0};
        double_complex prefac =std::pow(i,l)*std::pow(-1.0,m+0.5);
        if (component==0) {   // j = l+1/2 : k=-j-0.5 == j=1/2 ; k=-1
            double nn = (l==lround(j-0.5)) ? -sqrt((j+m)/(2.0*j)) : sqrt((j-m+1)/(2*j+2));
            return prefac * g * nn *SphericalHarmonics(l,lround(m-0.5))(c);
//            return g/r * sqrt(double_complex((k + 0.5 - m)/(2.0*k + 1))) *SphericalHarmonics(k,lround(m-0.5))(c);
        } else if (component==1) {
            double nn = (l==lround(j-0.5)) ? sqrt((j-m)/(2.0*j)) : sqrt((j+m+1)/(2*j+2));
            return prefac * g * nn *SphericalHarmonics(l,lround(m+0.5))(c);
//            return -g/r * sgnk* sqrt(double_complex((k + 0.5 + m)/(2.0*k + 1))) *SphericalHarmonics(k,lround(m+0.5))(c);
        } else if (component==2) {
            double nn = (l==lround(j-0.5)) ? sqrt((j-m+1)/(2.0*j+2.0)) : sqrt((j+m)/(2.0*j));
            long ll=  (l==lround(j-0.5))  ? l+1 : l-1;
            return prefac * i*f * nn *SphericalHarmonics(ll,lround(m-0.5))(c);
//            return i*f/r * sqrt(double_complex((-k + 0.5 - m)/(-2.0*k + 1))) *SphericalHarmonics(-k,lround(m-0.5))(c);
        } else if (component==3) {
            double nn = (l==lround(j-0.5)) ? sqrt((j+m+1)/(2.0*j+2.0)) : -sqrt((j-m)/(2.0*j));
            long ll=  (l==lround(j-0.5))  ? l+1 : l-1;
            return prefac * i*f * nn *SphericalHarmonics(ll,lround(m+0.5))(c);
//            return -i*f/r * sgnk* sqrt(double_complex((-k + 0.5 - m)/(-2.0*k + 1))) *SphericalHarmonics(-k,lround(m+0.5))(c);
        }
        MADNESS_EXCEPTION("confused component in ExactSpinor::angular",1);
        return {0.0,0.0};
    }

    Spinor get_spinor(World& world) const {
        Spinor spinor;
        component=0;
        spinor.components[0]=complex_factory_3d(world).functor(*this);
        component=1;
        spinor.components[1]=complex_factory_3d(world).functor(*this);
        component=2;
        spinor.components[2]=complex_factory_3d(world).functor(*this);
        component=3;
        spinor.components[3]=complex_factory_3d(world).functor(*this);
        return spinor;
    }


};


template<typename AnsatzT>
void orthonormalize(std::vector<Spinor>& arg, const AnsatzT ansatz) {
    World& world=arg.front().world();

    double maxq;
    do {
        auto bra=ansatz.make_vbra(arg);
        ansatz.normalize(bra,arg);
        Tensor<double_complex> S=matrix_inner(bra,arg);
        Tensor<double_complex> Q = NemoBase::Q2(S);
        maxq=0.0;
        for (int i=0; i<Q.dim(0); ++i)
            for (int j=0; j<i; ++j)
                maxq = std::max(maxq,std::abs(Q(i,j)));
        arg = transform(world, arg, Q, true);
        truncate(arg);
    } while (maxq>0.01);

    auto bra=ansatz.make_vbra(arg);
}



template<typename ansatzT>
Spinor apply_bsh(ansatzT& ansatz, const MatrixOperator& Hd, const MatrixOperator& Hv, const MatrixOperator& metric,
                 const Spinor& spinor, const double energy) {
    World& world=spinor.world();
    timer t(world);

//    Hv.print("Hv in apply_bsh");

    const double mu=ansatz.mu(energy);
    double fac= transform_c ? 1.0 : alpha1*alpha1;
    print("energy, mu, bsh_prefac in bsh: ",energy, mu,fac);

    auto bra=ansatz.make_bra(spinor);
    if (debug) show_norms(bra,spinor,"spinor before vpsi");
    auto vpsi=-2.0*Hv(spinor).truncate();
    if (debug) show_norms(bra,vpsi,"norms of vpsi");
    if (debug) show_norms(ansatz.make_bra(vpsi),vpsi,"<vpsi | vpsi>");
    t.tag("Vpsi");

    auto g=BSHOperator<3>(world,mu,lo,FunctionDefaults<3>::get_thresh());

    auto gvpsi1=apply(world,g,vpsi.components);
    t.tag("GVpsi");

    auto gvpsi=Spinor(truncate(gvpsi1));
    if (debug) show_norms(ansatz.make_bra(gvpsi),gvpsi,"|| gvpsi ||");

    auto result1=Hd(gvpsi);
    if (debug) show_norms(ansatz.make_bra(result1),result1,"|| Hd(gvpsi) ||");

    auto result2=energy*metric(gvpsi);
    if (debug) show_norms(ansatz.make_bra(result2),result2,"|| energy*N*(gvpsi) ||");
    Spinor result=0.5*fac*(result1 + result2);

    if (debug) show_norms(ansatz.make_bra(result),result,"<result | result>");
    t.tag("HdGVpsi");

    return result.truncate();
}

template<typename AnsatzT>
std::vector<Spinor> iterate(const std::vector<Spinor>& input, const std::vector<double> energy, const AnsatzT& ansatz, const int maxiter) {

    World& world=input.front().world();
    spinorallocator alloc(world,input.size());
    XNonlinearSolver<std::vector<Spinor>,double_complex,spinorallocator> solver(alloc);
    solver.set_maxsub(3);
    solver.do_print=true;

    auto Hv=ansatz.make_Hv(world);
    auto Hd=ansatz.make_Hd(world);
    auto H=Hd+Hv;
    auto metric= transform_c ? N_metric() : Metric();
    metric.print();
    std::vector<Spinor> current=copy(input);
    orthonormalize(current,ansatz);
    for (int iter=0; iter<maxiter; ++iter) {
        if (iter<3) solver.clear_subspace();    // start KAIN after 3 iterations only
        double wall0=wall_time();
        print("\nIteration ",iter);
        orthonormalize(current,ansatz);
        std::vector<Spinor> newpsi;
        for (auto& n : current) n.print_norms("current");
        for (int i=0; i<current.size(); ++i) newpsi.push_back(apply_bsh(ansatz,Hd,Hv,metric,current[i],energy[i]));
        for (auto& n : newpsi) n.print_norms("newpsi");
        auto residual=truncate(current-newpsi);
        double res=0.0;
        for (const auto& r : residual) res+=norm2(world,r.components);
        newpsi=truncate(solver.update(current,residual,1.e-4,3));
        orthonormalize(newpsi,ansatz);
        auto bra=ansatz.make_vbra(newpsi);
        Tensor<double_complex> fock=matrix_inner(bra,H(newpsi));
        fock+=conj_transpose(fock);
        fock*=0.5;
        auto ovlp=matrix_inner(bra,newpsi);
        print("ovlp before orthonormalization");
        print(ovlp);
        newpsi=truncate(orthonormalize_fock(newpsi,bra,fock));
        bra=ansatz.make_vbra(newpsi);
        ovlp=matrix_inner(bra,newpsi);
        std::vector<double> energy_differences;
        for (int i=0; i<current.size(); ++i) {
            newpsi[i].plot("psi"+std::to_string(i)+"_iteration"+std::to_string(iter)+"_ansatz"+ansatz.filename());
            double en=real(inner(bra[i],H(newpsi[i])));
//            show_norms(bra,H(newpsi),"energy contributions");
            double el_energy=compute_electronic_energy(en);
            double exact_el_energy=compute_electronic_energy(energy[i]);
            double diff=(el_energy-exact_el_energy);
            energy_differences.push_back(diff);
            printf("energy, el. energy, exact el. energy, difference %12.8f %12.8f %12.8f %4.1e\n", en, el_energy,exact_el_energy,diff);
        }
        current=newpsi;
        double wall1=wall_time();
        printf("elapsed time in iteration %2d: %6.2f with error %4.1e \n",iter,wall1-wall0,res );
//        printf("elapsed time in iteration %2d: %6.2f with energy/diff %12.8f %.2e \n",iter,wall1-wall0,compute_electronic_energy(en),
//               compute_electronic_energy(en) - compute_electronic_energy(energy));
    }
    return current;
}

template<typename ansatzT>
void run(World& world, ansatzT ansatz, const int nuclear_charge, const commandlineparser& parser, const int nstates) {
    print(" running Ansatz ",ansatz.name(), " transform_c",transform_c, "shift",shift);

    double thresh=FunctionDefaults<3>::get_thresh();
    long tmode=FunctionDefaults<3>::get_truncate_mode();
    Nemo nemo(world,parser);
    if (world.rank()==0) nemo.get_param().print("dft","end");


    std::vector<Spinor> guesses;
    std::vector<double> energies;

//    guesses.push_back(guess);
//    energies.push_back(ansatz.energy());
    ExactSpinor psi1s_half=ExactSpinor(1,'S',0.5,nuclear_charge,0.5);
    ExactSpinor psi1s_mhalf=ExactSpinor(1,'S',0.5,nuclear_charge,-0.5);
    ExactSpinor psi2s_half=ExactSpinor(2,'S',0.5,nuclear_charge,0.5);
    ExactSpinor psi2s_mhalf=ExactSpinor(2,'S',0.5,nuclear_charge,-0.5);
    ExactSpinor psi2p1_half  =ExactSpinor(2,'P',0.5,nuclear_charge, 0.5);
    ExactSpinor psi2p1_mhalf =ExactSpinor(2,'P',0.5,nuclear_charge,-0.5);
    ExactSpinor psi2p2_thalf =ExactSpinor(2,'P',1.5,nuclear_charge, 1.5);
    ExactSpinor psi2p2_half  =ExactSpinor(2,'P',1.5,nuclear_charge, 0.5);
    ExactSpinor psi2p2_mhalf =ExactSpinor(2,'P',1.5,nuclear_charge,-0.5);
    ExactSpinor psi2p2_mthalf=ExactSpinor(2,'P',1.5,nuclear_charge,-1.5);
//    std::vector<ExactSpinor> states ={psi1s,psi2p};
    std::vector<ExactSpinor> states ={psi1s_half,psi1s_mhalf,       // 1S 1/2
                                      psi2s_half,psi2s_mhalf,       // 2S 1/2
                                      psi2p1_half,psi2p1_mhalf,     // 2P 1/2
                                      psi2p2_mhalf,psi2p2_thalf,psi2p2_mhalf,psi2p2_mthalf}; // 2P 3/2


    sgl_guess sgl_1s=sgl_guess(1,0,0,nuclear_charge);
    sgl_guess sgl_2s=sgl_guess(2,0,0,nuclear_charge);
    sgl_guess sgl_2p0=sgl_guess(2,1,0,nuclear_charge);
    sgl_guess sgl_2p1=sgl_guess(2,1,1,nuclear_charge);
    sgl_guess sgl_2pm1=sgl_guess(2,1,-1,nuclear_charge);
    sgl_guess sgl_3s=sgl_guess(3,0,0,nuclear_charge);
    std::vector<sgl_guess> sgl_states={sgl_1s,sgl_2s,sgl_2p0,sgl_2p1,sgl_2pm1,sgl_3s};
    std::vector<Spinor> guess;
    const bool nemoguess=true;
    print("\nUsing Schroedinger guess\n");
    if (nemoguess) {

        std::vector<complex_function_3d> wf;
        for (int i=0; i<nstates; ++i) wf.push_back(sgl_states[i].get_wf(world));
        guess= schrodinger2dirac(wf,ansatz,nuclear_charge);
    } else {
        for (int i=0; i<nstates; ++i) {
            states[i].set_ansatz(ansatz);
            guess.push_back(states[i].get_spinor(world));
            states[i].print();
            guess.back().print_norms("guess");

        }
    }
    orthonormalize(guess,ansatz);
    auto bra=ansatz.make_vbra(guess);
    Tensor<double_complex> S=matrix_inner(bra,guess);
    print("initial overlap after  orthonormalization");
    print(S);




//    for (ExactSpinor& state : states) {
    for (int i=0; i<nstates; ++i) {
        ExactSpinor& state=states[i];
        state.compute_F=true;
        state.cusp_a=ansatz.get_cusp_a();
//        Spinor guess=state.get_spinor(world);
        Spinor spinorguess=guess[i];
        guesses.push_back(spinorguess);
        energies.push_back(state.get_energy());
        ansatz.normalize(guesses.back());
    }


    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma= compute_gamma(nuclear_charge);
    double electronic_energy=gamma*c*c - c*c;
    double energy=ansatz.energy();

    auto Hv=ansatz.make_Hv(world);
    auto Hd=ansatz.make_Hd(world);
    auto H=Hv+Hd;
    Hv.print("Hamiltonian Hv");
    Hd.print("Hamiltonian Hd");
    H.print("Hamiltonian Hd+Hv");

    for (int i=0; i<0; i++) {
        auto guess=guesses[i];
        Spinor Hpsi = H(guess);
        auto bra=ansatz.make_bra(guess);
        double en=real(inner(bra,H(guess)));
        show_norms(bra,H(guess),"energy contributions");
        print("computed energy             ", en);
        print("computed electronic energy  ", compute_electronic_energy(en) );
        print("exact electronic energy     ", electronic_energy);
        print("energy difference           ", compute_electronic_energy(en) - electronic_energy);
        show_norms(bra,guess,"norms of guess before iterate");
    }
    auto result=iterate(guesses,energies,ansatz,30);

}

template<typename ansatzT>
void eigenvector_test(World& world, const ansatzT ansatz, ExactSpinor es) {
    print("=============================================================");
    print("Ansatz", ansatz.name());
    LProjector lproj(world);

    es.set_ansatz(ansatz);
    es.print();
    auto exactF = es.get_spinor(world);
    ansatz.normalize(exactF);
    exactF.print_norms("exactf normalized");
    exactF.plot("exact"+es.filename()+ansatz.filename());
    lproj.analyze(exactF,"ExactSpinor");

    auto exactF1 = ansatz.make_guess(world);
    ansatz.normalize(exactF1);
    lproj.analyze(exactF1,"ansatz.make_guess");
    exactF1.print_norms("make_guess normalized");
    auto diff1=exactF-exactF1;
    diff1.print_norms("exactf-make_guess normalized");

    Spinor spinor = copy(exactF);

    auto Hv = ansatz.make_Hv(world);
    auto Hd = ansatz.make_Hd(world);
    auto H = Hv + Hd;
    H.print("H");
    Hd.print("Hd");
    Hv.print("Hv");


//    MatrixOperator sl_matrix;
//    auto sl= make_Zrsl(world,ansatz.nuclear_charge);
//    sl_matrix.add_submatrix(0, 0, sl);
//    sl_matrix.add_submatrix(2, 2, sl);
//
//    MatrixOperator snsl_matrix;
//    auto snsl=make_snsl(world,ansatz.nuclear_charge);
//    snsl_matrix.add_submatrix(0, 2, snsl);
//    snsl_matrix.add_submatrix(2, 0, snsl);
//
//    auto snsl_spinor=snsl_matrix(spinor);
//    snsl_spinor.print_norms("snsl_spinor");
//    auto sl_spinor=sl_matrix(spinor);
//    sl_spinor.print_norms("sl_spinor");
//
//    MatrixOperator snZ_matrix= make_Hv_reg3_snZ(world,ansatz.nuclear_charge,ansatz.a);
//    auto snZ_spinor=snZ_matrix(spinor);
//    snZ_spinor.print_norms("snZ_spinor");

    if (0) {
        auto Rinv = ansatz.Rinv(world);
        Rinv.print("Rinv");
        es.compute_F=false;
        auto exact1=es.get_spinor(world);
        auto Rinvexact1=double_complex(0.0,1.0)*Rinv(exact1);
        ansatz.normalize(Rinvexact1);
        Rinvexact1.print_norms("Rinvexact");
        auto diff=Rinvexact1-spinor;
        diff.print_norms("F-Rinv(psi)");
    }

    Spinor bra = ansatz.make_bra(spinor);
    ansatz.normalize(bra, spinor);
    auto norms = norm2s(world, spinor.components);

    print("");
    auto Hdspinor = Hd(spinor);
    auto Hvspinor = Hv(spinor);
    auto Hspinor = H(spinor);
    auto hnorms = norm2s(world, Hspinor.components);
    auto energy_norms=norms;
    for (auto& c : energy_norms) c*=es.get_energy();
    print("E * component norms", energy_norms);
    print("H(spinor) component norms", hnorms);
    auto en = inner(bra, Hspinor);

    auto diff = Hspinor - en * spinor;
    spinor.print_norms("spinor");
    Hspinor.print_norms("Hspinor");
    Hdspinor.print_norms("Hdspinor");
    Hvspinor.print_norms("Hvspinor");
    diff.print_norms("diff_Hspinor_en_spinor");
    double c=1.0/alpha1;
    print("energy", en, real(en - c * c), "difference", real(en - c * c) - (es.get_energy() - c * c));print("");
    print("");
}

int main(int argc, char* argv[]) {
    World& world=initialize(argc,argv);
    if (world.rank()==0) {
        print("\n");
        print_centered("Dirac hydrogen atom");
    }
    startup(world,argc,argv,true);
    if (world.rank()==0) print(madness::info::print_revision_information());


    commandlineparser parser(argc,argv);
//    parser.set_keyval("dft","'k=8'");
    if (world.rank()==0) {
        print("\ncommand line parameters");
        parser.print_map();
    }

    // set defaults
    int nuclear_charge=92;
    int ansatz=3;
    double nemo_factor=1.3;
    FunctionDefaults<3>::set_cubic_cell(-20,20);
    FunctionDefaults<3>::set_k(12);
    FunctionDefaults<3>::set_thresh(1.e-10);
    int tmode=0;
    long nstates=1;
    if (parser.key_exists("charge")) nuclear_charge=atoi(parser.value("charge").c_str());
    if (parser.key_exists("k")) FunctionDefaults<3>::set_k(atoi(parser.value("k").c_str()));
    if (parser.key_exists("nstates")) nstates=atol(parser.value("nstates").c_str());
    if (parser.key_exists("thresh")) FunctionDefaults<3>::set_thresh(atof(parser.value("thresh").c_str()));
    if (parser.key_exists("L")) FunctionDefaults<3>::set_cubic_cell(atof(parser.value("L").c_str()),atof(parser.value("L").c_str()));
    if (parser.key_exists("transform_c")) transform_c=true;
    if (parser.key_exists("ansatz")) ansatz=atoi(parser.value("ansatz").c_str());
    if (parser.key_exists("nemo_factor")) nemo_factor=std::atof(parser.value("nemo_factor").c_str());
    if (parser.key_exists("use_ble")) use_ble=true;
    if (parser.key_exists("truncate_mode")) tmode=atoi(parser.value("truncate_mode").c_str());
    FunctionDefaults<3>::set_truncate_mode(tmode);

    print("\nCalculation parameters");
    print("thresh      ",FunctionDefaults<3>::get_thresh());
    print("k           ",FunctionDefaults<3>::get_k());
    print("trunc mode  ",FunctionDefaults<3>::get_truncate_mode());
    print("charge      ",nuclear_charge);
    print("cell        ",FunctionDefaults<3>::get_cell_width());
    print("transform_c ",transform_c);
    print("ues_ble     ",use_ble);


    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma= compute_gamma(nuclear_charge);
    print("speed of light",c);
    print("fine structure constant",alpha);
    const int k=1;
    print("gamma",gamma);
    double energy_exact=gamma*c*c - c*c;
    print("1s energy for Z=",nuclear_charge,": ",energy_exact);

    ExactSpinor es1s(1,'S',0.5,nuclear_charge);
    ExactSpinor es2s(2,'S',0.5,nuclear_charge);
    ExactSpinor es2p1(2,'P',0.5,nuclear_charge);
    ExactSpinor es2p2(2,'P',1.5,nuclear_charge);

    print("energies",es1s.get_energy()-c*c,
          es2s.get_energy()-c*c,
          es2p1.get_energy()-c*c,
          es2p2.get_energy()-c*c);

    {
        auto sglguess = sgl_guess(2, 1, 1, nuclear_charge);
        auto sgl_2s=sglguess.get_wf(world);
        std::vector<coord_3d> special_points(1,coord_3d({0.0,0.0,0.0}));
        real_function_3d pot=real_factory_3d(world)
                .functor([&nuclear_charge](const coord_3d& coord){return -nuclear_charge/(coord.normf()+epsilon);})
                .special_points(special_points);
        auto Epot=inner(sgl_2s,pot*sgl_2s);
        print("Epot",Epot);
        double_complex Ekin={0.0,0.0};
        for (int i=0; i<3; ++i) {
            complex_derivative_3d D(world,i);
            auto Dguess=D(sgl_2s);
            Ekin+=0.5*inner(Dguess,Dguess);
        }
        print("Ekin",Ekin);
        print("Etotal",Epot+Ekin);
        print("Eexact",sglguess.energy());
    }






//    eigenvector_test(world,Ansatz1(nuclear_charge,nemo_factor),ExactSpinor(1,'S',0.5,nuclear_charge, 0.5));
//    eigenvector_test(world,Ansatz1(nuclear_charge,nemo_factor),ExactSpinor(2,'S',0.5,nuclear_charge, 0.5));
//    eigenvector_test(world,Ansatz1(nuclear_charge,nemo_factor),ExactSpinor(2,'P',0.5,nuclear_charge, 0.5));
//    eigenvector_test(world,Ansatz1(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge, 1.5));
//    eigenvector_test(world,Ansatz0(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge, 0.5));
//    eigenvector_test(world,Ansatz0(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,-0.5));
//    eigenvector_test(world,Ansatz0(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,-1.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,nemo_factor),ExactSpinor(2,'S',0.5,nuclear_charge,-0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge, 1.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge, 0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,-0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,-1.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,nemo_factor),ExactSpinor(2,'P',0.5,nuclear_charge, 0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,nemo_factor),ExactSpinor(2,'P',0.5,nuclear_charge, 0.5));
//    eigenvector_test(world,Ansatz1(nuclear_charge,1),ExactSpinor(1,'S',0.5,nuclear_charge));
//    eigenvector_test(world,Ansatz2(nuclear_charge,1),ExactSpinor(1,'S',0.5,nuclear_charge));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,nemo_factor),ExactSpinor(1,'S',0.5,nuclear_charge));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,1.3),ExactSpinor(3,'D',2.5,nuclear_charge));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,nemo_factor),ExactSpinor(1,'S',0.5,nuclear_charge,0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,nemo_factor),ExactSpinor(1,'S',0.5,nuclear_charge,-0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,1.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,-0.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,nemo_factor),ExactSpinor(2,'P',1.5,nuclear_charge,-1.5));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,-1.2),ExactSpinor(2,'P',1.5,nuclear_charge));
//    eigenvector_test(world,Ansatz3(nuclear_charge,1,-1.2),ExactSpinor(1,'S',0.5,nuclear_charge));


    try {
        if (ansatz==0) run(world,Ansatz0(nuclear_charge,nemo_factor),nuclear_charge,parser,nstates);
        if (ansatz==1) run(world,Ansatz1(nuclear_charge,nemo_factor),nuclear_charge,parser,nstates);
        if (ansatz==2) run(world,Ansatz2(nuclear_charge,nemo_factor),nuclear_charge,parser,nstates);
        if (ansatz==3) run(world,Ansatz3(nuclear_charge,nemo_factor,1),nuclear_charge,parser,nstates);
    } catch (...) {
        std::cout << "caught an error " << std::endl;
    }
    finalize();
    return 0;


}
