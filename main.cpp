#include <iostream>
#define MADNESS_HAS_LIBXC 0
// #define USE_GENTENSOR 0 // only needed if madness was configured with `-D ENABLE_GENTENSOR=1
#include<madness.h>
#include<madness/chem.h>
#include<madness/world/timing_utilities.h>
using namespace madness;


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

    Spinor& truncate() {
        madness::truncate(components);
        return *this;
    }
    void normalize() {
        double_complex norm=inner(*this,*this);
        scale(world(),components,1.0/sqrt(real(norm)));
    }

    friend double_complex inner(const Spinor& bra, const Spinor& ket) {
        auto result= inner(bra.components,ket.components);  // implies complex conjugation
        return result;
    }

};

template<typename T>
Spinor operator*(const T fac, const Spinor& arg) {
    return Spinor(fac*arg.components);
}


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

    Spinor operator()(const Spinor& arg) const {
        World& world=arg.components[0].world();
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

    void print() const {
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
            .functor([&nuclear_charge](const coord_3d& r){return double_complex(-nuclear_charge/(r.normf()+1.e-10));});
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
    const double gamma=sqrt(1-nuclear_charge*nuclear_charge*alpha*alpha);

    /// !!! REGULARIZATION ERROR HERE IS CRITICAL !!!
    /// !!! KEEP EPSILON SMALL !!!
    double epsilon=1.e-10;

    complex_function_3d x_div_r2_f=complex_factory_3d(world).functor([&epsilon](const coord_3d& r){return r[0]/(inner(r,r)+epsilon);});
    complex_function_3d y_div_r2_f=complex_factory_3d(world).functor([&epsilon](const coord_3d& r){return r[1]/(inner(r,r)+epsilon);});
    complex_function_3d z_div_r2_f=complex_factory_3d(world).functor([&epsilon](const coord_3d& r){return r[2]/(inner(r,r)+epsilon);});

    auto x_div_r2=LocalPotentialOperator<double_complex,3>(world,"x/r2",x_div_r2_f);
    auto y_div_r2=LocalPotentialOperator<double_complex,3>(world,"y/r2",y_div_r2_f);
    auto z_div_r2=LocalPotentialOperator<double_complex,3>(world,"z/r2",z_div_r2_f);

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
    const double gamma=sqrt(1-nuclear_charge*nuclear_charge*alpha*alpha);

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

/// Hv for ansatz 3
MatrixOperator make_Hv_reg3(World& world, const int nuclear_charge, const double aa) {
    MatrixOperator H;
    const double_complex ii=double_complex(0.0,1.0);
    const double_complex one=double_complex(1.0,0.0);
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma=sqrt(1-nuclear_charge*nuclear_charge*alpha*alpha);

    complex_function_3d c2=complex_factory_3d(world)
            .functor([&c,&gamma](const coord_3d& r){return double_complex((gamma-1.0)*c*c,0.0);});
    auto V1=LocalPotentialOperator<double_complex,3>(world,"(gamma-1)c2",c2);
    return make_Hdiag(world,V1);

    /// !!! REGULARIZATION ERROR HERE IS CRITICAL !!!
    /// !!! KEEP EPSILON SMALL !!!
    double epsilon=1.e-8;
    double a=aa;
    print("a in make_Hv_reg3",a);

    // without the exponential factor we have in the (1,2) matrix:
    // i c \sn Z
    // with the exponential factor we have in the (1,2) matrix:
    // i c \sn (Z + \Sigma_U)
    // with \Sigma_U = -aZ/(a-1) exp(-aZr)/R
    // thus
    // i c \sn Z ( 1- a/(a-1) exp(-aZr)/R)

    double Z=nuclear_charge;
    coord_3d sp{0.0,0.0,0.0};
    std::vector<coord_3d> special_points(1,sp);
    complex_function_3d x_div_r=complex_factory_3d(world).functor([&epsilon, &Z, &a](const coord_3d& xyz){
        double r=xyz.normf();
        double extraterm=1.0;
        if (a>0.0) extraterm-=a/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
        return xyz[0]/(r+epsilon)*extraterm;
    }).special_level(15).special_points(special_points);;
    complex_function_3d y_div_r=complex_factory_3d(world).functor([&epsilon, &Z, &a](const coord_3d& xyz){
        double r=xyz.normf();
        double extraterm=1.0;
        if (a>0.0) extraterm-=a/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
        return xyz[1]/(r+epsilon)*extraterm;
    }).special_level(15).special_points(special_points);;
    complex_function_3d z_div_r=complex_factory_3d(world).functor([&epsilon, &Z, &a](const coord_3d& xyz){
        double r=xyz.normf();
        double extraterm=1.0;
        if (a>0.0) extraterm-=a/(a-1.0)*exp(-a*Z*r)/(1.0+1.0/(a-1.0)*exp(-a*Z*r));
        return xyz[2]/(r+epsilon)*extraterm;
    }).special_level(15).special_points(special_points);

    std::string extraname = (a>0.0) ? " slater" : "";
    auto x_div_rexp=LocalPotentialOperator<double_complex,3>(world,"x/r"+extraname,x_div_r);
    auto y_div_rexp=LocalPotentialOperator<double_complex,3>(world,"y/r"+extraname,y_div_r);
    auto z_div_rexp=LocalPotentialOperator<double_complex,3>(world,"z/r"+extraname,z_div_r);

    std::string filename="sn_slater";
    coord_3d lo({0,0,-.3});
    coord_3d hi({0,0, .3});
    const int npt=3001;
    plot_line(filename.c_str(),npt,lo,hi,real(x_div_r),real(y_div_r),real(z_div_r));

    H.add_operator(0,2,  ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));
    H.add_operator(0,3,  ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(0,3,     Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));

    H.add_operator(1,2,  ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(1,2,    -Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));
    H.add_operator(1,3, -ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));

    H.add_operator(2,0, -ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));
    H.add_operator(2,1, -ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(2,1,    -Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));

    H.add_operator(3,0, -ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_rexp));
    H.add_operator(3,0,     Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_rexp));
    H.add_operator(3,1,  ii*Z*c,std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_rexp));

    return H;
}



struct AnsatzBase {
    virtual void normalize(Spinor& bra, Spinor& ket) const {
        double_complex norm2=inner(bra,ket);
        print("norm2 in normalize",norm2);
        double norm=sqrt(real(norm2));
        scale(bra.world(),bra.components,1.0/norm);
        scale(ket.world(),ket.components,1.0/norm);
    }
    virtual Spinor make_guess(World& world) const = 0;
    virtual Spinor make_bra(const Spinor& ket) const = 0;
};

struct Ansatz0 : public AnsatzBase {
public:
    double nuclear_charge, k;
    Ansatz0(const double nuclear_charge, const int k) : nuclear_charge(nuclear_charge), k(k) {
        MADNESS_ASSERT(k==1);
    }
    const std::string name="0";
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double_complex one(1.0,0.0);
        const double n=1;
        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
        print("gamma-1",gamma-1.0);
        const double C=0.95*nuclear_charge/n;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*one*(1+gamma)*exp(-C*r.normf());});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*0.0*one;});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world, const int nuclear_charge) {
        return ::make_Hv(world,nuclear_charge);
    }

    Spinor make_bra(const Spinor& ket) const {
        return Spinor(copy(ket.world(),ket.components));
    }

    void normalize(Spinor& bra, Spinor& ket) const {
        ket.normalize();
        if (&bra!=&ket) bra.normalize();
    }
};

struct Ansatz1 : public AnsatzBase {
public:
    double nuclear_charge, k;
    Ansatz1(const double nuclear_charge, const int k) : nuclear_charge(nuclear_charge), k(k) {
        MADNESS_ASSERT(k==1);
    }
    const std::string name="1";
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;
        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
        const double C=0.95*nuclear_charge/n;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex((1+gamma)*exp(-C*r.normf()),0.0);});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex(0.0,0.0);});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world, const int nuclear_charge) {
        auto Hv=::make_Hv(world,nuclear_charge);
        Hv+= make_Hv_reg1(world,nuclear_charge);
        return Hv;
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double alpha=constants::fine_structure_constant;
        const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
        real_function_3d r2=real_factory_3d(world)
                .functor([&gamma](const coord_3d& r){return std::pow(r.normf(),2.0*(gamma-1));});
        Spinor result=Spinor(r2*ket.components);
        return result;
    }
};

struct Ansatz2 : public AnsatzBase {
public:
    double nuclear_charge, k;
    double a=1.2;
    Ansatz2(const double nuclear_charge, const int k) : nuclear_charge(nuclear_charge), k(k) {
        MADNESS_ASSERT(k==1);
    }
    const std::string name="2";
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;
        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
        const double C=0.95*nuclear_charge/n;
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex((1+gamma)*exp(-r.normf()),0.0);});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex(0.0,0.0)*exp(-r.normf());});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*r[2]/r.normf()*exp(-r.normf());});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-r.normf());});
        return result;
    }

    MatrixOperator make_Hv(World& world, const int nuclear_charge) {
        auto Hv=::make_Hv(world,nuclear_charge);
        Hv+= make_Hv_reg1(world,nuclear_charge);
        Hv+= make_Hv_reg2(world,nuclear_charge,a);
        return Hv;
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double alpha=constants::fine_structure_constant;
        const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
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
};

/// returns a (2,2) matrix
MatrixOperator make_sp(World& world) {
    MatrixOperator sp(2,2);
    const double_complex ii=double_complex(0.0,1.0);
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
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

struct Ansatz3 : public AnsatzBase {
public:
    double nuclear_charge, k;
    double a=1.3;
    Ansatz3(const double nuclear_charge, const int k) : nuclear_charge(nuclear_charge), k(k) {
        MADNESS_ASSERT(k==1);
    }
    const std::string name="3";
    Spinor make_guess(World& world) const {
        Spinor result;
        const double_complex ii(0.0,1.0);
        const double n=1;

        const double Z=double(nuclear_charge);
        const double alpha=constants::fine_structure_constant;
        const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
        const double C=0.95*nuclear_charge/n;
        print("C",C);
        result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex(exp(-r.normf()*r.normf()),0.0);});
        result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex(0.0,0.0);});
        result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return  double_complex(0.0,0.0);});
        result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return  double_complex(0.0,0.0);});
        return result;
    }


    MatrixOperator make_Hv(World& world, const int nuclear_charge) {
        auto Hv=::make_Hv_reg3(world,nuclear_charge,a);
        return Hv;
    }

    /// turns argument into its bra form: (r^(\gamma-1))^2
    Spinor make_bra(const Spinor& ket) const {
        World& world=ket.world();
        const double alpha=constants::fine_structure_constant;
        const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
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
};



/// this is c sigma p + beta m c^2

/// @param[in]  factor  scale the diagonal
MatrixOperator make_Hd(World& world, const double factor=1.0) {
    MatrixOperator Hd;
    MatrixOperator sp=make_sp(world);
    Hd.add_submatrix(0,2,sp);
    Hd.add_submatrix(2,0,sp);

    const double alpha=constants::fine_structure_constant;
    const double_complex c2=double_complex(1.0/(alpha*alpha),0.0)*factor;
    complex_function_3d V=complex_factory_3d(world).functor([](const coord_3d& r){return double_complex(1.0,0.0);});
    auto V1=LocalPotentialOperator<double_complex,3>(world,"mc2",V);
    Hd.add_operator(0,0, c2,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hd.add_operator(1,1, c2,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hd.add_operator(2,2,-c2,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hd.add_operator(3,3,-c2,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    return Hd;
}



Spinor apply_bsh(const MatrixOperator& Hd, const MatrixOperator& Hv, const Spinor& spinor, const double energy) {
    World& world=spinor.world();
    timer t(world);
    double lo=FunctionDefaults<3>::get_thresh();
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    double mu=sqrt(c*c - energy*energy/(c*c));
    print("energy in apply_bsh",energy);
    print("mu in bsh: ",mu);
    auto g=BSHOperator<3>(world,mu,1.e-8,FunctionDefaults<3>::get_thresh());
    auto vpsi=-2.0*Hv(spinor).truncate();
    t.tag("Vpsi");

    auto gvpsi1=apply(world,g,vpsi.components);
    t.tag("GVpsi");

    auto gvpsi=Spinor(truncate(gvpsi1));

    auto result=0.5*alpha*alpha*(Hd(gvpsi) + energy*gvpsi);
    t.tag("HdGVpsi");

    return result;
}

template<typename AnsatzT>
Spinor iterate(const MatrixOperator& Hv, Spinor input, const double energy_exact, const AnsatzT& ansatz, const int maxiter) {
    World& world=input.world();
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double nuclear_charge=ansatz.nuclear_charge;
    const double gamma=sqrt(1.0-nuclear_charge*nuclear_charge*alpha*alpha);

    coord_3d lo({0,0,-.3});
    coord_3d hi({0,0, .3});
    const int npt=3001;

//    double factor = (ansatz.name=="3")? gamma : 1.0 ;
    double factor = 1.0;
    auto Hd=make_Hd(world,factor);
    auto H=Hd+Hv;
    for (int i=0; i<maxiter; ++i) {
        double wall0=wall_time();
        print("\nIteration ",i);
        auto newpsi=apply_bsh(Hd,Hv,input,energy_exact+c*c);
        Spinor bra=ansatz.make_bra(newpsi);
        ansatz.normalize(bra,newpsi);
        std::string filename="spinor_0_re_ansatz"+ansatz.name+"_iter"+std::to_string(i);
        plot_line(filename.c_str(),npt,lo,hi,real(newpsi.components[0]));
        double_complex en=inner(bra,H(newpsi));
        print("computed energy  ", real(en)  - c * c);
        print("exact energy     ", energy_exact);
        print("energy difference", (real(en) - c * c) - energy_exact);
        input=newpsi;
        double wall1=wall_time();
        printf("elapsed time in iteration %6.2f\n",wall1-wall0);
    }
    return input;
}

template<typename ansatzT>
void run(World& world, ansatzT ansatz, const int nuclear_charge, const int k) {
    print(" running Ansatz ",ansatz.name);
    Spinor guess = ansatz.make_guess(world);
    Spinor bra=ansatz.make_bra(guess);
    ansatz.normalize(guess,guess);
    ansatz.normalize(guess,bra);
    ansatz.normalize(guess,bra);
    double n=real(inner(bra,guess));
    print("norm in the beginning",n);
    {
        auto norms = inner(world, bra.components, guess.components);
        print("norms", real(norms));
    }

    coord_3d lo({0,0,-.3});
    coord_3d hi({0,0, .3});
    const int npt=3001;
    std::string filename="spinor_0_re_ansatz"+ansatz.name+"_guess";
    plot_line(filename.c_str(),npt,lo,hi,real(guess.components[0]));

    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma=sqrt(1.0-nuclear_charge*nuclear_charge*alpha*alpha);
    double energy_exact=gamma*c*c - c*c;

    auto Hv=ansatz.make_Hv(world,nuclear_charge);
//    const double factor = ansatz.name=="3" ? gamma : 1.0;
    const double factor = 1.0;
    auto Hd=make_Hd(world,factor);
    auto H=Hv+Hd;
    H.print();
    Spinor Hpsi = H(guess);
    double_complex energy = inner(bra, Hpsi);
    print("computed energy  ", real(energy)  - c * c);
    print("exact energy     ", energy_exact);
    print("energy difference", (real(energy) - c * c) - energy_exact);

    auto result=iterate(Hv,guess,energy_exact,ansatz,10);

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

    print("\nCalculation parameters");
    print("thresh   ",FunctionDefaults<3>::get_thresh());
    print("k        ",FunctionDefaults<3>::get_k());
    print("charge   ",nuclear_charge);
    print("cell     ",FunctionDefaults<3>::get_cell_width());


    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma=sqrt(1-nuclear_charge*nuclear_charge*alpha*alpha);
    print("speed of light",c);
    print("fine structure constant",alpha);
    const int k=1;
    print("gamma",gamma);
    double energy_exact=gamma*c*c - c*c;
    print("1s energy for Z=",nuclear_charge,": ",energy_exact);
    coord_3d lo({0,0,-.1});
    coord_3d hi({0,0, .1});
    const int npt=3001;

    try {
//        run(world,Ansatz0(nuclear_charge,k),nuclear_charge,k);
//        run(world,Ansatz1(nuclear_charge,k),nuclear_charge,k);
//        run(world,Ansatz2(nuclear_charge,k),nuclear_charge,k);
        run(world,Ansatz3(nuclear_charge,k),nuclear_charge,k);
    } catch (...) {
        std::cout << "caught an error " << std::endl;
    }
    finalize();
    return 0;


}
