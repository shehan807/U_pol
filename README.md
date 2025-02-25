<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


# Polarization Energy, $U_{pol}(\{ \mathbf{r}_ {i} \} ,\{ \mathbf{d}_ {i} \} )$, Calculator

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#theory">Theory</a>
    </li>
    <li>
      <a href="#code-structure">Code Structure</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#future-work">Future Work</a></li>
  </ol>
</details>


<!-- Theory -->
## Theory

Polarization is defined by the redistribution of a particle's electronic density due to local electric fields. In the simplest case, the polarizability of two particles with polarizabilities $\alpha$ is proportional to $-\alpha^2/r^6$ (in an average sense). This is captured in many nonpolarizable molecular dynamics (MD) codes through the Lennard-Jones potential, where

$$U_{LJ}(r) = 4\epsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}-\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]$$

includes a long range attractive $r^{-6}$ dependence due to London dispersion. Moreover, partial atomic charges, $q_i$ also incorporate polarization in an *implicit* way, where charge values are ``enhanced'' say in the case of condensed phase systems. However, effective treatment of polarization neglect the fundamental dependence charge distribution has on a system's state and the dynamic response to fluctuations in the electric field upon sub-picosecond molecular motion (Rick and Stuart, 2002).

### Shell Models

One way to represent polarization in MD is by representing dipoles of finite length as a pair of point charges attached by a harmonic spring, e.g. "shell models" or otherwise referred to as "Drude oscillator models". For the sake of terminology, there is a subtle distinction between *shell* models, in which dipoles are treated adiabatically, and *Drude* models, where dipole oscillations are thermal, thereby giving rise to dispersion interaction. The basic schematic is provided below:

![alt text](http://localscf.com/localscf.com/images/drude.jpg)

There is a positive "core" charge located at the nucleus and a negative "shell" charge with fixed magnitude, $\pm q_{i}$, for some neutral atom site $i$ respectively. Charged species can be accomadated by inluding a permanent charge $z_{i}$ with the core (nuclear) charge. The dipole moment then determined by

$$ \mathbf{\mu }_ i = -q_i \mathbf{d}_ i$$

**The goal of this program is to determine the potential energy of the induced dipoles, $U_{ind}$, provided the initial positions of the atoms and Drude particles are given (i.e., input ``.cif``, ``.pdb``, etc.).** $U_{ind}$ is broken up into three components, (1) the polarization energy, $U_{pol}$, (2) the induced-dipole/induced-dipole interaction energy, $U_{\mu\mu}$, and (3)the interaction with any static field, $U_{stat}$:

$$U_{ind} = U_{pol} + U_{\mu\mu} + U_{stat}$$

The polarization energy is intuitive--it is the energy considering the harmonic spring between the core and shell charges,

$$U_{pol} = \frac{1}{2}\sum_{i=1}^{N} k_i d_i^2$$

where the spring constants $k_i$ can be found (for an isotropic shell model) through the polarizability, $\alpha_i = q_i^2 / k_i$.

The electrostatic interaction between independent polarizable atoms is written as the sum of the charge-charge interactions between all four charge sites):

$$U_{\mu\mu} = \frac{1}{2}\sum_{i=1}^{N}\sum_{j\neq i} q_iq_j \left[\frac{1}{|\mathbf{r}_ {ij}|}-\frac{1}{|\mathbf{r}_ {ij} - \mathbf{d}_ j|}-\frac{1}{|\mathbf{r}_ {ij} - \mathbf{d}_ i|}+\frac{1}{|\mathbf{r}_ {ij} - \mathbf{d}_ j + \mathbf{d}_ i|}\right]$$

Note that the Coulomb interactions between core and shell charges on the same site are typically excluded. Finally, the interaction of the induced dipoles with the static field is written as the sum,

$$U_{stat} = - \sum_{i=1}^{N} q_i \left[\mathbf{r}_ i \cdot \mathbf{E}_ i^0 - (\mathbf{r}_ i + \mathbf{d}_ i) \cdot \mathbf{E}_ i^{0\' }\right]$$

where $\mathbf{E}_ i^0$ and $\mathbf{E}_ i^{0\' }$ are the static fields at the core and shell charge sites, respectively. The static field at some site $i$ is determined by

$$\mathbf{E}_ i^0 = \sum_{j\neq i} \frac{q_i \mathbf{r}_ {ij}}{r_{ij}^3}$$

Of course, these equations are not without limitations relative to quantum mechanical theory. Namely, polarizable MD models that invoke the shell model depend on approximations of (1) representing the electronic charge density with point charges (or in other methods, dipoles), (2) treating electrostatic polarizabilities isotropically, and (3) terminating the electrostaic interactions after the dipole-dipole term.

<!-- CODE STRUCTURE -->
## Code Structure

**Task:** For some $\{\mathbf{r}_ i\}$, determine $U(\{\mathbf{r}_ {i}\},\{\mathbf{d}_ {i}^{\text{min} }\})$

### 1. Assign Drudes (if not otherwise included in initial structure file)

### 2. Evaluate Initial $U_{pol}(\{\mathbf{r}_ {i}\},\{\mathbf{d}_ {i}\})$

#### Including Thole screening

### 3. Minimize $U_{pol}(\{\mathbf{r}_ {i}\},\{\mathbf{d}_ {i}\})$ w.r.t. $\{\mathbf{d}_ i\}$

#### Obtaining gradients via JAXS

#### Iterative Methods

##### Conjugate Gradient
##### BFGS
