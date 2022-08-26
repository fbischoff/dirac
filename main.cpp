#include <iostream>
#define MADNESS_HAS_LIBXC 0
// #define USE_GENTENSOR 0 // only needed if madness was configured with `-D ENABLE_GENTENSOR=1
#include<madness.h>
#include<madness/chem.h>
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

    Spinor& operator+=(const Spinor& other) {
        components+=other.components;
        return *this;
    }

    void normalize() {
        double_complex norm=inner(*this,*this);
        scale(world(),components,1.0/sqrt(real(norm)));
    }

    friend double_complex inner(const Spinor& bra, const Spinor& ket) {
        return inner(bra.components,ket.components);  // implies complex conjugation
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

Spinor make_guess(World& world, const int nuclear_charge, const int k) {
    Spinor result;
    MADNESS_ASSERT(k==1);
    const double_complex ii(0.0,1.0);
    const double_complex one(1.0,0.0);
    const double n=1;
    const double Z=double(nuclear_charge);
    const double alpha=constants::fine_structure_constant;
    const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
    print("gamma-1",gamma-1.0);
    const double C=nuclear_charge/n;
    result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*one*(1+gamma)*exp(-C*r.normf());});
    result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*0.0*one;});
    result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
    result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii,&one](const coord_3d& r){return std::pow(r.normf(),gamma-1.0)*ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
    return result;
}

Spinor make_guess_regularized(World& world, const int nuclear_charge, const int k) {
    Spinor result;
    MADNESS_ASSERT(k==1);
    const double_complex ii(0.0,1.0);
    const double n=1;
    const double Z=double(nuclear_charge);
    const double alpha=constants::fine_structure_constant;
    const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
    const double C=nuclear_charge/n;
    result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex((1+gamma)*exp(-C*r.normf()),0.0);});
    result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return double_complex(0.0,0.0);});
    result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
    result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
    return result;
}
Spinor make_guess_regularized_bra(World& world, const int nuclear_charge, const int k) {
    Spinor result;
    MADNESS_ASSERT(k==1);
    const double_complex ii(0.0,1.0);
    const double n=1;
    const double Z=double(nuclear_charge);
    const double alpha=constants::fine_structure_constant;
    const double gamma=sqrt(k*k-nuclear_charge*nuclear_charge*alpha*alpha);
    const double C=nuclear_charge/n;
    result.components[0]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return std::pow(r.normf(),(gamma-1.0)*2.0)*double_complex((1+gamma)*exp(-C*r.normf()),0.0);});
    result.components[1]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return std::pow(r.normf(),(gamma-1.0)*2.0)*double_complex(0.0,0.0);});
    result.components[2]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return std::pow(r.normf(),(gamma-1.0)*2.0)*ii*Z*alpha*r[2]/r.normf()*exp(-C*r.normf());});
    result.components[3]=complex_factory_3d(world).functor([&Z,&gamma,&alpha,&C,&ii](const coord_3d& r){return std::pow(r.normf(),(gamma-1.0)*2.0)*ii*Z*alpha*(r[0] + ii*r[1])/r.normf()*exp(-C*r.normf());});
    return result;
}

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

/// this is c sigma p + beta m c^2
MatrixOperator make_Hd(World& world) {
    MatrixOperator Hd;
    MatrixOperator sp=make_sp(world);
    Hd.add_submatrix(0,2,sp);
    Hd.add_submatrix(2,0,sp);
    const double alpha=constants::fine_structure_constant;
    complex_function_3d mc2=complex_factory_3d(world).functor([&alpha](const coord_3d& r){return double_complex(1.0/(alpha*alpha),0.0);});
    auto vmc2=LocalPotentialOperator<double_complex,3>(world,"mc^2",mc2);
    Hd.add_operator(0,0, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(vmc2));
    Hd.add_operator(1,1, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(vmc2));
    Hd.add_operator(2,2,-1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(vmc2));
    Hd.add_operator(3,3,-1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(vmc2));
    return Hd;
}

/// this is the nuclear potential on the diagonal
MatrixOperator make_Hv(World& world, const int nuclear_charge) {
    MatrixOperator Hv;
    double Z=double (nuclear_charge);
    complex_function_3d V=complex_factory_3d(world).functor([&Z](const coord_3d& r){return double_complex(-Z/(r.normf()+1.e-12),0.0);});
    auto V1=LocalPotentialOperator<double_complex,3>(world,"V",V);
    Hv.add_operator(0,0, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hv.add_operator(1,1, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hv.add_operator(2,2, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    Hv.add_operator(3,3, 1.0,std::make_shared<LocalPotentialOperator<double_complex,3>>(V1));
    return Hv;
}

MatrixOperator make_H(World& world, const int nuclear_charge, const int regularization_order) {
    MatrixOperator Hd=make_Hd(world);
    Hd+=make_Hv(world,nuclear_charge);
    const double_complex ii=double_complex(0.0,1.0);
    const double_complex one=double_complex(1.0,0.0);
    const double Z=double(nuclear_charge);
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const double gamma=sqrt(1-nuclear_charge*nuclear_charge*alpha*alpha);
    complex_function_3d x_div_r2_f=complex_factory_3d(world).functor([&gamma,&c,&ii](const coord_3d& r){return r[0]/(inner(r,r)+1.e-12);});
    complex_function_3d y_div_r2_f=complex_factory_3d(world).functor([&gamma,&c,&ii](const coord_3d& r){return r[1]/(inner(r,r)+1.e-12);});
    complex_function_3d z_div_r2_f=complex_factory_3d(world).functor([&gamma,&c,&ii](const coord_3d& r){return r[2]/(inner(r,r)+1.e-12);});
    complex_function_3d zero1=complex_factory_3d(world).functor([&alpha](const coord_3d& r){return double_complex(0.0,0.0);});

    auto x_div_r2=LocalPotentialOperator<double_complex,3>(world,"x/r2",x_div_r2_f);
    auto y_div_r2=LocalPotentialOperator<double_complex,3>(world,"y/r2",y_div_r2_f);
    auto z_div_r2=LocalPotentialOperator<double_complex,3>(world,"z/r2",z_div_r2_f);


    if (regularization_order==1) {
        Hd.add_operator(0,2,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));
        Hd.add_operator(0,3,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
        Hd.add_operator(0,3,-one*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));
    }

    if (regularization_order==1) {
        Hd.add_operator(1,2,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
        Hd.add_operator(1,2,    c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));
        Hd.add_operator(1,3, ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));
    }

    if (regularization_order==1) {
        Hd.add_operator(2,0,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));
        Hd.add_operator(2,1,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
        Hd.add_operator(2,1,-one*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));
    }

    if (regularization_order==1) {
        Hd.add_operator(3,0,-ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(x_div_r2));
        Hd.add_operator(3,0,    c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(y_div_r2));
        Hd.add_operator(3,1, ii*c*(gamma-1),std::make_shared<LocalPotentialOperator<double_complex,3>>(z_div_r2));
    }

    return Hd;
}


Spinor apply_bsh(const MatrixOperator& H, const Spinor& spinor, const double energy) {
    World& world=spinor.world();
    double lo=FunctionDefaults<3>::get_thresh();
    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    double mu=sqrt(c*c - energy*energy/c*c);
    auto g=BSHOperator<3>(world,mu,lo,FunctionDefaults<3>::get_thresh());
    auto Hd= make_Hd(world);

}



int main(int argc, char* argv[]) {
    World& world=initialize(argc,argv);
    startup(world,argc,argv,true);
    FunctionDefaults<3>::set_cubic_cell(-20,20);
    FunctionDefaults<3>::set_k(12);
    FunctionDefaults<3>::set_thresh(1.e-10);
    print("Dirac hydrogen atom");

    const double alpha=constants::fine_structure_constant;
    const double c=1.0/alpha;
    const int nuclear_charge=92;
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
        {
            Spinor guess = make_guess(world, nuclear_charge, k);
            plot_line("spinor_0_re",npt,lo,hi,real(guess.components[0]));
            MatrixOperator Hd = make_H(world, nuclear_charge,0);
            Hd.print();
            double_complex norm2 = inner(guess, guess);
            print("norm2 of guess", norm2);
            Spinor Hpsi = Hd(guess);
            double_complex energy = inner(guess, Hpsi);
            print("unnormalized energy", energy);
            print("computed energy  ", real(energy) / real(norm2) - c * c);
            print("exact energy     ", energy_exact);
            print("energy difference", (real(energy) / real(norm2) - c * c) - energy_exact);
        }

        {
            print("\n\nregularized version\n\n");
            Spinor guess = make_guess_regularized(world, nuclear_charge, k);
            plot_line("spinor_0_re_regularized",npt,lo,hi,real(guess.components[0]));
            Spinor guess_bra = make_guess_regularized_bra(world, nuclear_charge, k);
            MatrixOperator Hd = make_H(world, nuclear_charge,1);
            Hd.print();
            double_complex norm2 = inner(guess_bra, guess);
            print("norm2 of guess", norm2);
            Spinor Hpsi = Hd(guess);
            double_complex energy = inner(guess_bra, Hpsi);
            print("unnormalized energy", energy);
            print("computed energy  ", real(energy) / real(norm2) - c * c);
            print("exact energy     ", energy_exact);
            print("energy difference", (real(energy) / real(norm2) - c * c) - energy_exact);
        }


    } catch (...) {
        std::cout << "caught an error " << std::endl;
    }
    finalize();
    return 0;


}
