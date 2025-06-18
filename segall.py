we want to make a proper software for advanced geoscience solutions for carbon and hydrogen storage using streamlit. 

so the software name is stratagraph. 

in the website. there are four modules (four versions) so far now. 

1. VHydro: Qualitative Classification [Hydrocarbon Potential](version 1.0) inspired by CORA graph dataset: I named it VHydro it reflects the below my paper novel graph dataset. 

title: A Graph Convolutional Network Approach to Qualitative Classification of Hydrocarbon Zones Using Petrophysical Properties in Well Logs. author is me (venkateshwaran)
abstract: The discovery of hydrocarbon reserves has declined significantly in recent years owing to the structural and lithological heterogeneities present in reservoirs. To overcome this decline, it is crucial to incorporate advanced computational methods, such as machine learning (ML) and deep learning (DL). These technologies can facilitate more precise discovery of hydrocarbon reserves, thereby replenishing and increasing the supply of proven reserves. By utilizing ML and DL, likelihood of errors arising from human error or bias during exploration activities can be diminished substantially. This is due to the extensive incorporation of sophisticated statistical techniques within the applications of ML and DL. Well logs provide valuable information about the physical characteristics of subterranean fluids and rocks. In the McKee field in New Zealand, the lithology of three wells was determined using petrophysical parameters, such as porosity, permeability, water saturation, volume shale, and oil saturation that were extracted from the well logs. The unsupervised method K-means clustering (KMC) was used to perform facies classification tasks, utilizing clusters that matched the well’s facies and ranged from 5 to 10 and each well yielded six pairs of outputs. Graph convolutional networks (GCN) are reliable technique for working with graph representations. The best performance is achieved by directly integrating graph convolutions with feature information and related parameters. The petrophysical parameters were combined with the unlabeled KMC outputs to form the GCN dataset. The initial potential zones were classified into five classes based on petrophysical criteria: very high, high, moderate, low, and very low. This GCN approach was used to identify each graph quality in the dataset. The hydrocarbon potential of the three wells was evaluated using the graph dataset and the GCN approach, which produced results with higher accuracy when real labels were used. The findings of this study indicate that the identification of a hydrocarbon-rich region through the utilization of a graph that integrates lithological and petrophysical data requires a comprehensive understanding of the subsurface that goes beyond lithology alone. To achieve this, a novel method for predicting hydrocarbon potential based on GCN is proposed, which combines graph datasets derived from well logs consisting of petrophysical entities and depth values.

user needs to input well logs, it will take to the further pages like petrophysical and facies classification. so facies and petrophysical entities together the graph dataset made and to classify the potential hydrocarbons. the final procedure in this module. users can directly input the facies and petrophysical properties then go to the prediction. same like this module user can input the instead well log they can easily use the geomechanical properties and facies to predict caprock potential rather than hydrocarbon potential.


2. DIFac: Inter-facies Interactions [Litho and Mechanical Stratigraphy] version 2.0: i named it like interfacies interactions which is inspired by reddit graph dataset. This module research paper is under progress.

idea: facies based on the lithology but when the potential related to the mechanical ability the geomechanical based strata classification require like mechanical stratigraphy (1D Mechanical Earth Model) and facies classification vertically can classify multiple facies zones like in the first facies first zone (because first facies clusterings are 30 from the full well means, each clusterings composes of 3, 4, 10 sequence of depths). same like mechanical stratigraphy also its facies represents the same but it varies its not depend on the lithology so its based on the mechanical properties. So we will interconnect the both facies together based on the clusterings in the lithostratigraphy first facies first clusters have 5 sequence of depths and mechanical stratigraphy have 15 sequence of depths. after connecting those clusters, from the mechanical facies depths  check the remaining 10 sequence depths and its associated mechanical facies like edges to influence the other lithology facies  mechanical properties. the influences will predicted by GNN models.


3. MechBrt: Seismic Inversion + Well log data based Caprock Potential [Layer-wise Brittleness Index] this is purely under conceptual idea only. now little progress achieved. Research Paper under progress. there is no inspiration to any dataset. pure novel

idea: the prestack seismic inversion have the each traces (time domain) and well log (depth domain but can convert it to time). whatever, still people are using synthetic data and static model dataset to assume the reservoir properties in a field scale from cm scale (well logs) so that case the seismic inversion comes into play it performed well and the accuracy is far better than the traditional one. But the inversion related jobs cannot be validated without well logs is key source. scientist and engineers are spending too much time on interpreting those. so best case integrating together like layer wise. prestack seismic inversion results have the young's modulus and poisson's ration to calculate the brittleness index and same like well logs we have the empirical formulas to calculate the same brittleness index. the target is brittleness index (BI). we will classify the potential based on BI like very high to very low. nodes are seismic trace values in each layers and same layers well log depths also. to drive it into the GraphSAGE++ models. this idea can be enhanced by using pointwise instead of the layers wise and zone wise. or others. 

MechSim: Geomechanical Modeling [PINNs and Data-driven & vertical and horizontal displacement model] 

Idea: this is full physics informed neural network and data-driven approach. like user can upload the input and output from the geomechanical simulations. because there is no available UI to train their results for the data-driven models for the future predictions like res net and U-Net. like that

main idea: is to model the segall model poroelasticity below. this model can be use in the GNN simulations like we are connecting the MechBrt model like layer wise because they also used the layer wise approaximations. to predict and tune it further to simulate the model using GNN based on segall and other models as well.

################################################
# Calculation of ground deformations due to depletion and injection in a confined reservoir

By D. N. Espinoza ([DNEGeomechanics@twitter](https://twitter.com/DNEGeomechanics), [dnicolasespinoza@github](https://github.com/dnicolasespinoza), espinoza@austin.utexas.edu)

## 1. Introduction

This is an interactive plot of ground deformations due to depletion/injection from "Earthquakes triggered by fluid extraction" by Segall, Geology 1989. doi: [10.1130/0091-7613(1989)017](https://pubs.geoscienceworld.org/gsa/geology/article-abstract/17/10/942/186508/Earthquakes-triggered-by-fluid-extraction?redirectedFrom=fulltext)

## 2. Model

The assumed model is plane strain with poroelasticity in the confined reservoir and elasticity in the surroundings:

```
___________________________       |x
                    |             |____ y
                    |D    
          _______   |_      
         |_______|  |T     
              ---         
               a        
                      

```
where
* $D$: depth to reservoir top
* $T$: reservoir thickness
* $a$: reservoir half length

The predicted displacements and strains are:

$u_x(x=0,y,t) = \frac{2(1+\nu_u)BT \Delta m(t)}{3\pi \rho_o} \left[ \tan^{-1}(\xi_-) - \tan^{-1}(\xi_+)\right]$

$u_y(x=0,y,t) = \frac{2(1+\nu_u)BT \Delta m(t)}{6\pi \rho_o} \log \left[ \frac{1+\xi_+^2}{1-\xi_-^2}\right]$

$\epsilon_{yy}(x=0,y,t) = \frac{2(1+\nu_u)BT \Delta m(t)}{3\pi \rho_o D} \left[ \frac{\xi_+}{1+\xi_+^2} - \frac{\xi_-}{1+\xi_-^2}\right]$

where
* $\nu_u$: undrained Poisson ratio
* $B$: Skempton parameter
* $\Delta m(t)= \frac{(1-2\nu) \alpha \rho_o}{2\mu (1+\nu)} \left[ \sigma_{kk} + \frac{3}{B}p \right]$: change of fluid mass content per unit of volume
* $\rho_o$: fluid initial mass density
* $\xi_- = (y-a)/D$
* $\xi_+ = (y+a)/D$

Note: signs changed for $u_y$ and $\epsilon_{yy}$ in code to match Fig. 5 in paper.'''

## 3. Code
@interact(D=(100,3000,25),a=(100,1000,25),T=(25,300,25),Deltamt=(-10,10,1)) # Widget variables
def plotter(D=1000,a=500,T=100,Deltamt=5):
    B = 0.9     # Skempton parameter
    PRu = 0.3   # Undrained Poisson Ratio
    rho0 = 1000 # Fluid density [kg/m3]
    #T = 100     # Reservoir thickness [m]
    #D = 500    # Reservoir depth [m]
    #a = 500     # Reservoir half length [m]
    # change of fluid mass per unit volume
    #Deltamt = 1 # Fluid extraction/injection [check units, equation 1b]

    # define linear space for y
    y = np.linspace(-D*6,D*6,100) # y-coordinate (horizontal) [m]
    zetap = (y+a)/D
    zetam = (y-a)/D

    ux_norm =    1 * (np.arctan(zetam) - np.arctan(zetap))        # [-] Vert. disp norm
    uy_norm =    -1/2* (np.log(1+zetap**2) - np.log(1+zetam**2))  # [-] Hz. disp norm
    epsyy_norm = -1 * (zetap/(1+zetap**2) - zetam/(1+zetam**2))   # [-] Hz. strain norm

    ux = 2*(1+PRu)*B*T*Deltamt / (3*3.1415*rho0) * ux_norm          # [m] Vert. disp
    uy = 2*(1+PRu)*B*T*Deltamt / (3*3.1415*rho0) * uy_norm          # [m] Hz. disp
    epsyy = 2*(1+PRu)*B*T*Deltamt / (3*3.1415*rho0*D) * epsyy_norm  # [m] Hz. strain

    ### plotting
    plt.subplot(311) #  plot domain
    plt.plot([-3000,3000],[0,0], 'k-') #ground
    plt.plot([-a,a,a,-a,-a],[-D,-D,-D-T,-D-T,-D], 'k-') #reservoir
    #plt.plot(y/D,epsyy_norm, 'g-', label = "epsyy")
    # plotting options
    plt.xlim(-3000,3000)
    plt.ylim(-3000,100)
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')

    plt.subplot(312) # plot normalized values, compare to Fig. 5 in Segall's 1989 paper
    plt.plot(y/D,uy_norm, 'b-', label = "uy_norm")
    plt.plot(y/D,ux_norm, 'r-', label = "ux_norm")
    plt.plot(y/D,epsyy_norm, 'g-', label = "epsyy_norm")
    # plotting options
    plt.xlabel('y/D [-]')
    plt.ylabel('[-]')
    #plt.xlim(0,360)
    #plt.ylim(-5000,15000)
    plt.legend()

    plt.subplot(313) #  plot actual values
    plt.plot(y,uy, 'b-', label = "uy: Hz disp.")
    plt.plot(y,ux, 'r-', label = "ux: Vert disp.")
    #plt.plot(y/D,epsyy_norm, 'g-', label = "epsyy")
    # plotting options
    plt.xlabel('y [-]')
    plt.ylabel('Displacements [m]')
    plt.xlim(-3000,3000)
    plt.ylim(-0.15,0.15)
    plt.legend()
    plt.show()

################################################

