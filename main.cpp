#include <iostream>
#define MADNESS_HAS_LIBXC 0
// #define USE_GENTENSOR 0 // only needed if madness was configured with `-D ENABLE_GENTENSOR=1
#include<madness.h>
#include<madness/chem.h>
#include<madness/mra/nonlinsol.h>
#include<madness/world/timing_utilities.h>
using namespace madness;


static bool transform_c=false;
static bool debug=false;
static double alpha1=constants::fine_structure_constant;
static double shift=0.0;
static bool exact_has_singularity=false;


/// returns the complex value of a given spherical harmonic
struct SphericalHarmonics{
    int l, m;
    bool zero=false;
    SphericalHarmonics(const int l, const int m) : l(l), m(m) {
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
        if ((l==0) and (m== 0)) return 0.5*sqrt(1.0/constants::pi);

        if ((l==1) and (m==-1)) return 0.5*sqrt(1.5/constants::pi) * (x - i*y)/r;
        if ((l==1) and (m== 0)) return 0.5*sqrt(3.0/constants::pi) * z/r;
        if ((l==1) and (m== 1)) return -0.5*sqrt(1.5/constants::pi) * (x + i*y)/r;

        if ((l==2) and (m==-2)) return  0.25*sqrt(7.5/constants::pi) * std::pow((x - i*y)/r,2.0);
        if ((l==2) and (m==-1)) return  0.5 *sqrt(7.5/constants::pi) * (x - i*y)*z/r2;
        if ((l==2) and (m== 0)) return  0.25*sqrt(5.0/constants::pi) * (3.0*z*z - r2)/r2;
        if ((l==2) and (m== 1)) return -0.5 *sqrt(7.5/constants::pi) * (x + i*y)*z/r2;
        if ((l==2) and (m== 2)) return  0.25*sqrt(7.5/constants::pi) * std::pow((x + i*y)/r,2.0);
        MADNESS_EXCEPTION("out of range in SphericalHarmonics",1);

        return double_complex(0.0,0.0);
    }

};

double compute_gamma(const double nuclear_charge) {
    return sqrt(1-nuclear_charge*nuclear_charge*alpha1*alpha1);
}
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

    std::string info() const {return "D"+std::to_string(axis);}

    functionT operator()(const functionT& ket) const {
        vecfuncT vket(1,ket);
        return this->operator()(vket)[0];
    }

    vecfuncT operator()(const vecfuncT& vket) const {
        auto gradop = free_space_derivative<T,NDIM>(world, axis);
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

// The default constructor for functions does not initialize
// them to any value, but the solver needs functions initialized
// to zero for which we also need the world object.
struct allocator {
    World& world;
//    const int n;
//
    /// @param[in]	world	the world
    /// @param[in]	nn		the number of functions in a given vector
    allocator(World& world) : world(world) {
    }

    /// allocate a vector of n empty functions
   Spinor operator()() {
        return Spinor(world);
    }
};

/// class defining an operator in matrix form, fixed to size (4,4)
class MatrixOperator {
public:
    typedef std::vector<std::pair<double_complex,std::shared_ptr<SCFOperatorBase<double_complex,3>>>> opT;

    MatrixOperator(const int i=4, const int j=4) {
        elements.resize(i);
        for (auto& e : elements) e.resize(j);
    }

    int nrow() const {return elements.size();}
    int ncol() const {
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
                for (auto op : ops) {
                    auto fac=op.first;
                    result.components[i]+=fac * (*op.second)(arg.components[j]);
                }
            }
        }
        double_complex norm2=inner(arg,arg);

        if (std::abs(norm1-norm2)/std::abs(norm1)>1.e-10) throw;

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

    void pad(World& world) {
        complex_function_3d zero1=complex_factory_3d(world)
                .functor([&](const coord_3d& r){return double_complex(0.0,0.0);});
        auto zero=LocalPotentialOperator<double_complex,3>(world,"0",zero1);
        for (int i=0; i<ncol(); ++i) {
            for (int j=0; j<nrow(); ++j) {
                if (elements[i][j].empty())
                    add_operator(i,j,double_complex(0.0,0.0),std::make_shared<LocalPotentialOperator<double_complex,3>>(zero));
            }
        }
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

Metric inverse(const Metric& other) {
    Metric result;
    result.c0=1.0/other.c0;
    result.c1=1.0/other.c1;
    result.c2=1.0/other.c2;
    result.c3=1.0/other.c3;
    return result;
}

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
Metric I() {
    return Metric();
}


void show_norms(const Spinor& bra, const Spinor& ket, std::string name) {
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
MatrixOperator make_Hv(World& world, const int nuclear_charge) {
    complex_function_3d V=complex_factory_3d(world)
            .functor([&nuclear_charge](const coord_3d& r){return double_complex(-nuclear_charge/(r.normf()+1.e-13));});
    auto V1=LocalPotentialOperator<double_complex,3>(world,"V",V);
    return make_Hdiag(world,V1);
}

/// Hv for ansatz 1
MatrixOperator make_Hv_reg1(World& world, const int nuclear_charge) {
    MatrixOperator H;
    const double_complex ii=double_complex(0.0,1.0);
    const double_complex one=double_complex(1.0,0.0);
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;

    /// !!! REGULARIZATION ERROR HERE IS CRITICAL !!!
    /// !!! KEEP EPSILON SMALL !!!
    double epsilon=1.e-10;

    complex_function_3d x_div_r2_f=complex_factory_3d(world).functor([&epsilon](const coord_3d& r){return r[0]/(inner(r,r)+epsilon);});
    complex_function_3d y_div_r2_f=complex_factory_3d(world).functor([&epsilon](const coord_3d& r){return r[1]/(inner(r,r)+epsilon);});
    complex_function_3d z_div_r2_f=complex_factory_3d(world).functor([&epsilon](const coord_3d& r){return r[2]/(inner(r,r)+epsilon);});

    auto x_div_r2=LocalPotentialOperator<double_complex,3>(world,"x/r2",x_div_r2_f);
    auto y_div_r2=LocalPotentialOperator<double_complex,3>(world,"y/r2",y_div_r2_f);
    auto z_div_r2=LocalPotentialOperator<double_complex,3>(world,"z/r2",z_div_r2_f);

    double gamma=compute_gamma(nuclear_charge);
    H.add_operator(0,2,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));
    H.add_operator(0,3,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
    H.add_operator(0,3,-one*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));

    H.add_operator(1,2,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
    H.add_operator(1,2,    c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));
    H.add_operator(1,3, ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));

    H.add_operator(2,0,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));
    H.add_operator(2,1,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
    H.add_operator(2,1,-one*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));

    H.add_operator(3,0,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
    H.add_operator(3,0,    c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));
    H.add_operator(3,1, ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));
    return H;
}

/// Hv for ansatz 2
MatrixOperator make_Hv_reg2(World& world, const int nuclear_charge, const double a) {
    MatrixOperator H;
    const double_complex ii=double_complex(0.0,1.0);
    const double_complex one=double_complex(1.0,0.0);
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    double gamma=compute_gamma(nuclear_charge);

    /// !!! REGULARIZATION ERROR HERE IS CRITICAL !!!
    /// !!! KEEP EPSILON SMALL !!!
    double epsilon=1.e-8;

    double Z=nuclear_charge;
    complex_function_3d x_div_r_exp=complex_factory_3d(world).functor([&epsilon, &a, &Z](const coord_3d& xyz){
        double r=xyz.normf();
        return xyz[0]/(r+epsilon)*(-a*Z)/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
    });
    complex_function_3d y_div_r_exp=complex_factory_3d(world).functor([&epsilon, &a, &Z](const coord_3d& xyz){
        double r=xyz.normf();
        return xyz[1]/(r+epsilon)*(-a*Z)/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
    });
    complex_function_3d z_div_r_exp=complex_factory_3d(world).functor([&epsilon, &a, &Z](const coord_3d& xyz){
        double r=xyz.normf();
        return xyz[2]/(r+epsilon)*(-a*Z)/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
    });

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

/// returns a (4,4) hermition matrix with H=i \vec \alpha \vec {xyz}
MatrixOperator make_alpha_sn(World& world, const LocalPotentialOperator<double_complex, 3>& x,
                             const LocalPotentialOperator<double_complex, 3>& y,
                             const LocalPotentialOperator<double_complex, 3>& z) {
    MatrixOperator H(4,4);
    const double_complex ii=double_complex(0.0,1.0);
    const double_complex one=double_complex(1.0,0.0);

    H.add_operator(0, 2,  ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));
    H.add_operator(0, 3,  ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
    H.add_operator(0, 3, one, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));

    H.add_operator(1, 2,  ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
    H.add_operator(1, 2,-one, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));
    H.add_operator(1, 3, -ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));

    H.add_operator(2, 0, -ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));
    H.add_operator(2, 1, -ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
    H.add_operator(2, 1,-one, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));

    H.add_operator(3, 0, -ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(x));
    H.add_operator(3, 0, one, std::make_shared<LocalPotentialOperator<double_complex, 3>>(y));
    H.add_operator(3, 1,  ii, std::make_shared<LocalPotentialOperator<double_complex, 3>>(z));
    return H;
}


/// the off-diagonal blocks of ansatz 3
MatrixOperator make_Hv_reg3_snZ(World& world, const int nuclear_charge, const double aa, const bool longrange_correction) {

    /// !!! REGULARIZATION ERROR HERE IS CRITICAL !!!
    /// !!! KEEP EPSILON SMALL !!!
    print("a in make_Hv_reg3", aa);

    const double_complex ii = double_complex(0.0, 1.0);
    const double_complex one = double_complex(1.0, 0.0);

    // without the exponential factor we have in the (1,2) matrix:
    // i c \sn Z
    // with the exponential factor we have in the (1,2) matrix:
    // i c \sn (Z + \Sigma_U)
    // with \Sigma_U = -aZ/(a-1) exp(-aZr)/R
    // thus
    // i c \sn Z ( 1- a/(a-1) exp(-aZr)/R)
    /// !!! REGULARIZATION ERROR HERE IS CRITICAL !!!
    /// !!! KEEP EPSILON SMALL !!!
    double epsilon=1.e-8;
    double c=1/alpha1;

    double a=aa;
    double Z = nuclear_charge;
    coord_3d sp{0.0, 0.0, 0.0};
    std::vector<coord_3d> special_points(1, sp);

    int axis=0;
    bool lr_correction=longrange_correction;
    auto func =[&epsilon, &Z, &a, &c, &lr_correction, &axis](const coord_3d& xyz) {
        double r = xyz.normf();
        double gamma= compute_gamma(Z);
        double sigmaterms = 1.0;
        if (a > 0.0) sigmaterms -= a / (a - 1.0) * exp(-a * Z * r) / (1.0 + 1.0 / (a - 1.0) * exp(-a * Z * r));
//      if (lr_correction) sigmaterms+=(-2.0*c*r*(std::pow(r,gamma)-r) - (gamma-1)*exp(c*r*r) + gamma-1)/(r*(exp(c*r*r)-1.0 + std::pow(r,gamma)));
        double cc=3.0;
        if (lr_correction) sigmaterms+=(-2.0*cc*r*(std::pow(r,gamma)-r)*exp(-cc*r*r) - (gamma-1)+ (gamma-1)*exp(-cc*r*r))/(r+(-1.0 + std::pow(r,gamma))*exp(-cc*r*r));
        return Z * c * xyz[axis] / (r + epsilon) * sigmaterms;
    };

    std::string filename1="lr_correction";
    coord_3d lo1={-20,0,0};
    coord_3d hi1={20,0,0};

    plot_line(world,filename1.c_str(),100,lo1,hi1,func);
    plot_plane<3>(world,func,filename1);
    real_function_3d bla = real_factory_3d(world).functor(func);
    double norm=bla.norm2();
    print("norm2 of bla",norm);

    axis=0;
    complex_function_3d x_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);;
    axis=1;
    complex_function_3d y_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);;
    axis=2;
    complex_function_3d z_div_r = complex_factory_3d(world).functor(func).special_level(15).special_points(special_points);;


    std::string extraname = (a > 0.0) ? " slater" : "";
    if (lr_correction) extraname+=" lr_corr";
    auto x_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "x/r" + extraname, x_div_r);
    auto y_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "y/r" + extraname, y_div_r);
    auto z_div_rexp = LocalPotentialOperator<double_complex, 3>(world, "z/r" + extraname, z_div_r);

    std::string filename = "sn_slater";
    coord_3d lo({0, 0, -3});
    coord_3d hi({0, 0, 3});
    const int npt = 3001;
    plot_line(filename.c_str(), npt, lo, hi, real(x_div_r), real(y_div_r), real(z_div_r));
    plot_plane(world, real(x_div_r), real(y_div_r), real(z_div_r), filename);

    return make_alpha_sn(world,x_div_rexp, y_div_rexp, z_div_rexp);
}

/// Hv for ansatz 3
MatrixOperator make_Hv_reg3_version1(World& world, const int nuclear_charge, const double aa, const bool lr_correction) {
    MatrixOperator H=make_Hv_reg3_snZ(world,nuclear_charge,aa,lr_correction);

    double gamma = compute_gamma(nuclear_charge);
    double c=1.0/alpha1;

    complex_function_3d V = complex_factory_3d(world).functor(
            [](const coord_3d& r) { return double_complex(1.0, 0.0); });
    auto V1 = LocalPotentialOperator<double_complex, 3>(world, "(gamma-1)c2", V);
    auto V2 = LocalPotentialOperator<double_complex, 3>(world, "-(gamma-1)c2", V);
    H.add_operator(0, 0, (gamma - 1.0) * c * c, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
    H.add_operator(1, 1, (gamma - 1.0) * c * c, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V1));
    H.add_operator(2, 2, -(gamma - 1.0) * c * c, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V2));
    H.add_operator(3, 3, -(gamma - 1.0) * c * c, std::make_shared<LocalPotentialOperator<double_complex, 3>>(V2));
    return H;
}

/// Hv for ansatz 3
MatrixOperator make_Hv_reg3_version3(World& world, const int nuclear_charge, const double aa, const bool lr_correction) {
    MatrixOperator H=make_Hv_reg3_snZ(world,nuclear_charge,aa,lr_correction);

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
MatrixOperator make_Hd(World& world, std::pair<double_complex,std::string> ll,
                       std::pair<double_complex,std::string> ss) {
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


struct ExactSpinor {
    int n, k, l;
    mutable int component=0;
    double E, C, gamma, Z, j, m;
    ExactSpinor(const int n, const char lc, const double j, const int Z)
        : ExactSpinor(n, l_char_to_int(lc),j,Z) { }
    ExactSpinor(const int n, const int l, const double j, const int Z)
        : n(n), l(l), j(j), Z(Z) {
        if (std::abs(j-(l+0.5))<1.e-10) k=lround(-j-0.5);       // j = l+1/2
        else k=lround(j+0.5);
        MADNESS_ASSERT(n==-k);

        m=j;
        gamma=sqrt(k*k - Z*Z*alpha1*alpha1);
        E=gamma/n /(alpha1*alpha1);
        C=Z/n;
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
        madness::print("energy = ",E, E-1.0/(alpha1*alpha1));
    }


    double_complex operator()(const coord_3d& c) const {
        double r=c.normf();
        double rho=2*C*r;
        double radial=exp(-rho*0.5);
        if (exact_has_singularity) radial*=std::pow(rho,gamma);
        int mm=m;
        int kk=(k>0) ? k : -k-1;
        double g=(n+gamma)*radial;
        double f=Z*alpha1*radial;
        double_complex i={0.0,1.0};
        double sgnk= (k>0) ? 1.0 : -1.0;


        MADNESS_CHECK((l==lround(j-0.5)));
        double_complex ii =std::pow(i,l)*std::pow(-1.0,m+0.5);
        if (component==0) {   // j = l+1/2 : k=-j-0.5 == j=1/2 ; k=-1
            double nn = (l==lround(j-0.5)) ? -sqrt((j+m)/(2.0*j)) : sqrt((j-m+1)/(2*j+2));
            return ii * g/r * nn *SphericalHarmonics(l,lround(m-0.5))(c);
//            return g/r * sqrt(double_complex((k + 0.5 - m)/(2.0*k + 1))) *SphericalHarmonics(k,lround(m-0.5))(c);
        } else if (component==1) {
            double nn = (l==lround(j-0.5)) ? sqrt((j-m)/(2.0*j)) : sqrt((j+m+1)/(2*j+2));
            return ii * g/r * nn *SphericalHarmonics(l,lround(m+0.5))(c);
//            return -g/r * sgnk* sqrt(double_complex((k + 0.5 + m)/(2.0*k + 1))) *SphericalHarmonics(k,lround(m+0.5))(c);
        } else if (component==2) {
            double nn = (l==lround(j-0.5)) ? sqrt((j-m+1)/(2.0*j+2.0)) : sqrt((j+m)/(2.0*j));
            int ll=  (l==lround(j-0.5))  ? l+1 : l-1;
            return -ii * i*f/r * nn *SphericalHarmonics(ll,lround(m-0.5))(c);
//            return i*f/r * sqrt(double_complex((-k + 0.5 - m)/(-2.0*k + 1))) *SphericalHarmonics(-k,lround(m-0.5))(c);
        } else if (component==3) {
            double nn = (l==lround(j-0.5)) ? sqrt((j+m+1)/(2.0*j+2.0)) : -sqrt((j-m)/(2.0*j));
            int ll=  (l==lround(j-0.5))  ? l+1 : l-1;
            return ii * i*f/r * nn *SphericalHarmonics(ll,lround(m+0.5))(c);
//            return -i*f/r * sgnk* sqrt(double_complex((-k + 0.5 - m)/(-2.0*k + 1))) *SphericalHarmonics(-k,lround(m+0.5))(c);
        }
        MADNESS_EXCEPTION("confused component in ExactSpinor",1);
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

struct AnsatzBase {
    virtual std::string filename() const {return this->name(); }
    virtual std::string name() const =0;

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
    virtual Spinor make_guess(World& world) const = 0;
    virtual MatrixOperator make_Hd(World& world) const = 0;
    virtual MatrixOperator R(World& world) const {
        MADNESS_EXCEPTION("no R implemented in this ansatz",1);
    }
    virtual MatrixOperator Rinv(World& world) const {
        MADNESS_EXCEPTION("no Rinv implemented in this ansatz",1);
    }
    virtual Spinor make_bra(const Spinor& ket) const = 0;
    virtual double mu(const double energy) const = 0;
};

struct Ansatz0 : public AnsatzBase {
public:
    double nuclear_charge, k;
    Ansatz0(const double nuclear_charge, const int k) : nuclear_charge(nuclear_charge), k(k) {
        MADNESS_ASSERT(k==1);
    }
    std::string name() const {
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
        print("gamma-1",gamma-1.0);
        const double C=nuclear_charge/n;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*one*(1+gamma)*exp(-C*r.normf());});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*0.0*one;});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world, const int nuclear_charge) const {
        return ::make_Hv(world,nuclear_charge);
    }

    Spinor make_bra(const Spinor& ket) const {
        return Spinor(copy(ket.world(),ket.components));
    }
    MatrixOperator make_Hd(World& world) const {
        double c2=1.0/(alpha1*alpha1);
        return ::make_Hd(world,{c2,"mc2"},{-c2,"-mc2"});
    }

    double mu(const double energy) const {
        return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
    }
    double energy() const {
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }

    MatrixOperator R(World& world) const {
        MADNESS_CHECK(!exact_has_singularity);
        complex_function_3d one1=complex_factory_3d(world).functor([](const coord_3d& r) {return double_complex(1.0,0.0);});
        auto one = LocalPotentialOperator<double_complex, 3>(world, "1" , one1);
        return make_Hdiag(world,one);
    }
    MatrixOperator Rinv(World& world) const {
        MADNESS_CHECK(!exact_has_singularity);
        complex_function_3d one1=complex_factory_3d(world).functor([](const coord_3d& r) {return double_complex(1.0,0.0);});
        auto one = LocalPotentialOperator<double_complex, 3>(world, "1" , one1);
        return make_Hdiag(world,one);
    }

};

struct Ansatz1 : public AnsatzBase {
public:
    double nuclear_charge, k;
    Ansatz1(const double nuclear_charge, const int k) : nuclear_charge(nuclear_charge), k(k) {
        MADNESS_ASSERT(k==1);
    }
    std::string name() const {
        return "1";
    }
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;
        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma= compute_gamma(nuclear_charge);
        const double C=nuclear_charge/n;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex((1+gamma)*exp(-C*r.normf()),0.0);});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex(0.0,0.0);});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world, const int nuclear_charge) const {
        auto Hv=::make_Hv(world,nuclear_charge);
        Hv+= make_Hv_reg1(world,nuclear_charge);
        return Hv;
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double gamma=compute_gamma(nuclear_charge);
        real_function_3d r2=real_factory_3d(world)
                .functor([&gamma](const coord_3d& r){return std::pow(r.normf(),2.0*(gamma-1));});
        Spinor result=Spinor(r2*ket.components);
        return result;
    }
    MatrixOperator make_Hd(World& world) const {
        double c2=1.0/(alpha1*alpha1);
        return ::make_Hd(world,{c2,"mc2"},{-c2,"-mc2"});
    }
    double mu(const double energy) const {
        return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
    }
    double energy() const {
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }
//    MatrixOperator R(World& world) const {
//        const double gamma=compute_gamma(nuclear_charge);
//        complex_function_3d r1=complex_factory_3d(world)
//                .functor([&gamma](const coord_3d& r){return std::pow(r.normf(),(gamma-1));});
//        auto r = LocalPotentialOperator<double_complex, 3>(world, "R" , r1);
//        return make_Hdiag(world,r);
//    }
    MatrixOperator Rinv(World& world) const {
        const double gamma=compute_gamma(nuclear_charge);

        auto func_with_singularity= [&gamma](const coord_3d& r){return std::pow(r.normf(),-(gamma-1));};
        auto func_without_singularity= [&gamma](const coord_3d& r){return 1.0;};
        complex_function_3d r1= exact_has_singularity ? complex_factory_3d(world).functor(func_with_singularity)
                : complex_factory_3d(world).functor(func_without_singularity);
        auto r = LocalPotentialOperator<double_complex, 3>(world, "Rinv" , r1);
        return make_Hdiag(world,r);
    }
};

struct Ansatz2 : public AnsatzBase {
public:
    double nuclear_charge, k;
    double a=1.2;
    Ansatz2(const double nuclear_charge, const int k) : nuclear_charge(nuclear_charge), k(k) {
        MADNESS_ASSERT(k==1);
    }
    std::string name() const {
        return "2";
    }
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;
        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma= compute_gamma(nuclear_charge);
        const double C=0.95*nuclear_charge/n;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex((1+gamma)*exp(-r.normf()),0.0);});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex(0.0,0.0)*exp(-r.normf());});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*r[2]/r.normf()*exp(-r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world, const int nuclear_charge) const {
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
    double mu(const double energy) const {
        return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
    }
    double energy() const {
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }
//    MatrixOperator R(World& world) const {
//        const double gamma=compute_gamma(nuclear_charge);
//        const double aa=a;
//        const double Z=nuclear_charge;
//        auto ncf_cusp = [&Z,&aa](const coord_3d& r) {
//            return 1.0+(1.0/(aa-1.0)*exp(-aa*Z*r.normf()));
//        };
//        auto ncf_singularity = [&gamma](const coord_3d& r) {
//            return std::pow(r.normf(),(gamma-1));
//        };
//        complex_function_3d r1=complex_factory_3d(world)
//                .functor([&ncf_cusp,&ncf_singularity](const coord_3d& r){return ncf_cusp(r)*ncf_singularity(r);});
//        auto r = LocalPotentialOperator<double_complex, 3>(world, "R" , r1);
//        return make_Hdiag(world,r);

//        double_complex ii={0.0,1.0};
//
//        double_complex fac=-ii /nuclear_charge/alpha1*(gamma-1);
//
//        complex_function_3d x1=complex_factory_3d(world).functor([&gamma,&fac](const coord_3d& r){return fac*r[0]*std::pow(r.normf(),(gamma-2));});
//        complex_function_3d y1=complex_factory_3d(world).functor([&gamma,&fac](const coord_3d& r){return fac*r[1]*std::pow(r.normf(),(gamma-2));});
//        complex_function_3d z1=complex_factory_3d(world).functor([&gamma,&fac](const coord_3d& r){return fac*r[2]*std::pow(r.normf(),(gamma-2));});
//
//        auto x = LocalPotentialOperator<double_complex, 3>(world, "Rx" , x1);
//        auto y = LocalPotentialOperator<double_complex, 3>(world, "Ry" , y1);
//        auto z = LocalPotentialOperator<double_complex, 3>(world, "Rz" , z1);
//        R+=make_alpha_sn(world,x,y,z);
//        return R;
//    }

    MatrixOperator Rinv(World& world) const {
        const double gamma=compute_gamma(nuclear_charge);
        const double aa=a;
        const double Z=nuclear_charge;
        auto ncf_cusp = [&Z,&aa](const coord_3d& r) {
            return 1.0+(1.0/(aa-1.0))*exp(-aa*Z*r.normf());
        };
        auto ncf_singularity = [&gamma](const coord_3d& r) {
            return std::pow(r.normf(),(gamma-1));
        };
        complex_function_3d r1= exact_has_singularity
                ? complex_factory_3d(world).functor([&ncf_cusp,&ncf_singularity](const coord_3d& r){return 1.0/(ncf_cusp(r)*ncf_singularity(r));})
                : complex_factory_3d(world).functor([&ncf_cusp,&ncf_singularity](const coord_3d& r){return 1.0/ncf_cusp(r);});
        double n1=r1.norm2();
        print("norm in Rinv",n1);
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
    double nuclear_charge;
    double a=1.3;
    int version=3;
    bool longrange_correction=true;

    std::string name() const {
        std::string v;
        if (version==1) v=", version 1, no transform, no shift, a="+std::to_string(a);
        if (version==2) v=", version 2, no transform, partition with Hv diagonal elements zero, a="+std::to_string(a);
        if (version==3) v=", version 3, shift by gamma c^2, then ST, a="+std::to_string(a);
        return std::string("3")+v;
    }
    std::string filename() const {
        return "v"+std::to_string(version) +"_a" +std::to_string(a);
    }

    Ansatz3(const double nuclear_charge, const int version, const double a, const bool longrange_correction) : nuclear_charge(nuclear_charge),
            version(version), a(a), longrange_correction(longrange_correction) {
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
        const double C=0.85*nuclear_charge/n;


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
                .special_level(30).special_points(special_points).thresh(1.e-7);

        for (int i=0; i<counter; ++i) bla=bla.square();
        double nn=bla.norm2();
        print("norm of real(guess)",nn);
        result.components[0]=convert<double,double_complex>(bla);
        result.components[1]=convert<double,double_complex>(zero);
        result.components[2]=0.01*convert<double,double_complex>(bla);
        result.components[3]=convert<double,double_complex>(zero);

        double norm=norm2(world,result.components);
        print("norm of guess ",norm);
        return result;
    }


    MatrixOperator make_Hv(World& world, const int nuclear_charge) const {
        if (version==1)  return ::make_Hv_reg3_version1(world,nuclear_charge,a,longrange_correction);
        if (version==2)  return ::make_Hv_reg3_snZ(world,nuclear_charge,a,longrange_correction);
        if (version==3)  return ::make_Hv_reg3_version3(world,nuclear_charge,a,longrange_correction);
        MADNESS_EXCEPTION("no version in ansatz 3 given",1);
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double alpha=constants::fine_structure_constant;
        const double gamma= compute_gamma(nuclear_charge);
        const double n=1;
        const double C=nuclear_charge/n;
        const double Z=nuclear_charge;
        const double aa=a;
        coord_3d sp{0.0,0.0,0.0};
        std::vector<coord_3d> special_points(1,sp);
        real_function_3d r2=real_factory_3d(world)
                .functor([&gamma, &Z, &aa](const coord_3d& r){
                    double R=std::pow(r.normf(),(gamma-1));
                    if (aa>0.0) R=R*(1.0+ 1.0/(aa-1.0)*exp(-aa*Z*r.normf()));
//                    double R=std::pow(r.normf(),(gamma-1)) * (1.0+1.0/(aa-1.0)*  exp(-aa*C*r.normf()));
                    return 2.0*R*R/(gamma+1.0);
                })
                .special_level(15).special_points(special_points);
        Spinor result=Spinor(r2*ket.components);
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

    double mu(const double energy) const {
        double gamma= compute_gamma(nuclear_charge);
        if (version==1) return sqrt(-energy*energy*alpha1*alpha1 + 1.0/(alpha1*alpha1));
        if (version==2) return sqrt(-energy*energy*alpha1*alpha1 + gamma*gamma/(alpha1*alpha1));
        if (version==3) return sqrt(energy*energy*alpha1*alpha1);
        MADNESS_EXCEPTION("no version in ansatz 3 given",1);
    }
    double energy() const {
        double gamma= compute_gamma(nuclear_charge);
        if (version==1) return gamma/(alpha1*alpha1);
        if (version==2) return gamma/(alpha1*alpha1);
        if (version==3) return compute_electronic_energy(gamma/(alpha1*alpha1));
        MADNESS_EXCEPTION("no version in ansatz 3 given",1);
        return compute_gamma(nuclear_charge)/(alpha1*alpha1);
    }
};



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

    auto g=BSHOperator<3>(world,mu,1.e-8,FunctionDefaults<3>::get_thresh());

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
Spinor iterate(const Spinor& input, const double energy, const AnsatzT& ansatz, const int maxiter) {
    World& world=input.world();
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double nuclear_charge=ansatz.nuclear_charge;
    const double gamma= compute_gamma(nuclear_charge);

    coord_3d lo({0,0,-3});
    coord_3d hi({0,0, 3});
    const int npt=3001;

    allocator alloc(world);
    XNonlinearSolver<Spinor,double_complex,allocator> solver(alloc);
    solver.set_maxsub(5);
    solver.do_print=true;
    auto Hv=ansatz.make_Hv(world,nuclear_charge);
    auto Hd=ansatz.make_Hd(world);
    auto H=Hd+Hv;
    auto metric= transform_c ? N_metric() : Metric();
    metric.print();
    auto current=copy(input);
    ansatz.normalize(current);
    auto bra=ansatz.make_bra(current);
    if (debug) show_norms(bra,current,"current in iterate 1");
    for (int i=0; i<maxiter; ++i) {
        double wall0=wall_time();
        print("\nIteration ",i);
        auto newpsi=apply_bsh(ansatz,Hd,Hv,metric,current,energy);
        auto residual=current-newpsi;
        newpsi=solver.update(current,residual,1.e-4,100).truncate();
        auto res_bra=ansatz.make_bra(residual);
        if (debug) show_norms(res_bra,residual,"residual");
        Spinor bra=ansatz.make_bra(newpsi);
        ansatz.normalize(bra,newpsi);
        if (debug) show_norms(bra,newpsi,"newpsi after normalization");
        std::string filename="spinor_0_re_ansatz"+ansatz.filename()+"_iter"+std::to_string(i);
        plot_line(filename.c_str(),npt,lo,hi,real(newpsi.components[0]),real(newpsi.components[1]),real(newpsi.components[2]),real(newpsi.components[3]));
        plot_plane(world,real(newpsi.components),filename);
        filename="spinor_0_im_ansatz"+ansatz.filename()+"_iter"+std::to_string(i);
        plot_line(filename.c_str(),npt,lo,hi,imag(newpsi.components[0]),imag(newpsi.components[1]),imag(newpsi.components[2]),imag(newpsi.components[3]));
        plot_plane(world,imag(newpsi.components),filename);
        double en=real(inner(bra,H(newpsi)));
        if (debug) show_norms(bra,H(newpsi),"energy contributions");
        print("computed energy             ", en);
        print("computed electronic energy  ", compute_electronic_energy(en) );
        print("exact electronic energy     ", compute_electronic_energy(energy));
        print("energy difference           ", compute_electronic_energy(en) - compute_electronic_energy(energy));
        current=newpsi;
        double wall1=wall_time();
        printf("elapsed time in iteration %2d: %6.2f with energy/diff %12.8f %.2e \n",i,wall1-wall0,compute_electronic_energy(en),
               compute_electronic_energy(en) - compute_electronic_energy(energy));
    }
    return current;
}

template<typename ansatzT>
void run(World& world, ansatzT ansatz, const int nuclear_charge, const int k) {
    print(" running Ansatz ",ansatz.name(), " transform_c",transform_c, "shift",shift);
    Spinor guess = ansatz.make_guess(world);
    ansatz.normalize(guess);
    Spinor bra=ansatz.make_bra(guess);
    ansatz.normalize(guess);
    ansatz.normalize(guess);
    show_norms(bra,guess,"norms in the beginning");
    if (debug) show_norms(guess,bra,"norms in the beginning, hc");
//    guess+=guess;
//    guess.show_norms(bra,"norms in the beginning");
//    Metric N=N_metric();
//    guess=N(guess);
//    guess.show_norms(bra,"norms in the beginning");
    auto dipole_x=moments(world,0,1);
    auto dipole_y=moments(world,1,1);
    auto dipole_z=moments(world,2,1);

    auto quadrupole_x=moments(world,0,2);
    auto quadrupole_y=moments(world,1,2);
    auto quadrupole_z=moments(world,2,2);

    if (debug) show_norms(bra,dipole_x(guess),"dipole_x");
    if (debug) show_norms(bra,dipole_y(guess),"dipole_y");
    if (debug) show_norms(bra,dipole_z(guess),"dipole_z");
    if (debug) show_norms(bra,quadrupole_x(guess),"quadrupole_x");
    if (debug) show_norms(bra,quadrupole_y(guess),"quadrupole_y");
    if (debug) show_norms(bra,quadrupole_z(guess),"quadrupole_z");

    coord_3d lo({0,0,-.3});
    coord_3d hi({0,0, .3});
    const int npt=3001;
    std::string filename="spinor_0_re_ansatz"+ansatz.filename()+"_guess";
    plot_line(filename.c_str(),npt,lo,hi,real(guess.components[0]));
    plot_plane(world,real(guess.components),filename);
    filename="spinor_0_im_ansatz"+ansatz.filename()+"_guess";
    plot_plane(world,imag(guess.components),filename);

    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma= compute_gamma(nuclear_charge);
    double electronic_energy=gamma*c*c - c*c;
    double energy=ansatz.energy();

    auto Hv=ansatz.make_Hv(world,nuclear_charge);
    auto Hd=ansatz.make_Hd(world);
    auto H=Hv+Hd;
    Hv.print("Hamiltonian Hv");
    Hd.print("Hamiltonian Hd");
    H.print("Hamiltonian Hd+Hv");
    show_norms(bra,guess,"norms of guess before Hamiltonians");
    Spinor Hpsi = H(guess);

    if (debug) show_norms(bra,Hd(guess),"after <bra | Hd psi > ");
    if (debug) show_norms(bra,Hv(guess),"after <bra | Hv psi > ");
    if (debug) show_norms(bra,Hpsi,"after <bra | Hpsi > ");
    if (debug) show_norms(bra,guess,"norms of guess after Hamiltonians");
    double en=real(inner(bra,H(guess)));
    show_norms(bra,H(guess),"energy contributions");
    print("computed energy             ", en);
    print("computed electronic energy  ", compute_electronic_energy(en) );
    print("exact electronic energy     ", electronic_energy);
    print("energy difference           ", compute_electronic_energy(en) - electronic_energy);
    show_norms(bra,guess,"norms of guess before iterate");
    auto result=iterate(guess,energy,ansatz,2);

}

int main(int argc, char* argv[]) {
    World& world=initialize(argc,argv);
    if (world.rank()==0) {
        print("\n");
        print_centered("Dirac hydrogen atom");
    }
    startup(world,argc,argv,true);

    commandlineparser parser(argc,argv);
    if (world.rank()==0) {
        print("\ncommand line parameters");
        parser.print_map();
    }

    // set defaults
    int nuclear_charge=92;
    FunctionDefaults<3>::set_cubic_cell(-20,20);
    FunctionDefaults<3>::set_k(12);
    FunctionDefaults<3>::set_thresh(1.e-10);
    if (parser.key_exists("charge")) nuclear_charge=atoi(parser.value("charge").c_str());
    if (parser.key_exists("k")) FunctionDefaults<3>::set_k(atoi(parser.value("k").c_str()));
    if (parser.key_exists("thresh")) FunctionDefaults<3>::set_thresh(atof(parser.value("thresh").c_str()));
    if (parser.key_exists("L")) FunctionDefaults<3>::set_cubic_cell(atof(parser.value("L").c_str()),atof(parser.value("L").c_str()));
    if (parser.key_exists("transform_c")) transform_c=true;

    print("\nCalculation parameters");
    print("thresh      ",FunctionDefaults<3>::get_thresh());
    print("k           ",FunctionDefaults<3>::get_k());
    print("charge      ",nuclear_charge);
    print("cell        ",FunctionDefaults<3>::get_cell_width());
    print("transform_c ",transform_c);


    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma= compute_gamma(nuclear_charge);
    print("speed of light",c);
    print("fine structure constant",alpha);
    const int k=1;
    print("gamma",gamma);
    double energy_exact=gamma*c*c - c*c;
    print("1s energy for Z=",nuclear_charge,": ",energy_exact);
    coord_3d lo({0,0,-.1});
    coord_3d hi({0,0, .1});
    const int npt=3001;

    {
        exact_has_singularity=true;
        Ansatz2 ansatz(nuclear_charge,1);
        print("Ansatz:",ansatz.name());
        auto Hv=ansatz.make_Hv(world,nuclear_charge);
        auto Hd=ansatz.make_Hd(world);
        auto H=Hv+Hd;
        H.print("H");

        ExactSpinor es(2,'P',1.5,nuclear_charge);
        es.print();
        auto Rinv=ansatz.Rinv(world);
        auto exact=es.get_spinor(world);
        Spinor spinor=Rinv(exact);
//        Spinor spinor=es.get_spinor(world);
        Spinor bra=ansatz.make_bra(spinor);


        ansatz.normalize(bra,spinor);
        auto norms=norm2s(world,spinor.components);
        print("component norms",norms);
//        auto val=spinor.components[0]({0,1,1});
//        print("spinor(0,1,1)",val);
        auto n2=inner(spinor,spinor);
        print("norm2 ",n2);


        auto guess=ansatz.make_guess(world);
        auto guess_bra=ansatz.make_bra(guess);
        ansatz.normalize(guess_bra,guess);
        auto gnorms=norm2s(world,guess.components);
        print("guess component norms",gnorms);dd
        auto diff=guess-spinor;
        auto dnorms=norm2s(world,diff.components);
        print("difference component norms",dnorms);

        auto en1=inner(guess_bra,H(guess));
        print("guess energy",en1,en1-c*c, "difference",real(en1-c*c)-(es.get_energy()-c*c));
        auto en=inner(bra,H(spinor));
        print("energy",en,real(en-c*c), "difference", real(en-c*c)-(es.get_energy()-c*c));
        print("");
    }
    try {
        run(world,Ansatz0(nuclear_charge,k),nuclear_charge,k);
        run(world,Ansatz1(nuclear_charge,k),nuclear_charge,k);
//        run(world,Ansatz2(nuclear_charge,k),nuclear_charge,k);
        debug=true;
//        ansatz3_version=1;
//        transform_c=false;
//        transform_c=false;
//        run(world,Ansatz3(nuclear_charge,1,-1.3),nuclear_charge,k);
//        run(world,Ansatz3(nuclear_charge,1,1.1),nuclear_charge,k);
//        run(world,Ansatz3(nuclear_charge,1,1.3),nuclear_charge,k);
//        run(world,Ansatz3(nuclear_charge,1,1.5,false),nuclear_charge,k);
        run(world,Ansatz3(nuclear_charge,1,1.5,true),nuclear_charge,k);
        run(world,Ansatz3(nuclear_charge,1,2.0,false),nuclear_charge,k);
        run(world,Ansatz3(nuclear_charge,2,1.5,false),nuclear_charge,k);
//        run(world,Ansatz3(nuclear_charge,3),nuclear_charge,k);
    } catch (...) {
        std::cout << "caught an error " << std::endl;
    }
    finalize();
    return 0;


}
