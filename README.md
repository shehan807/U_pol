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


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Future Work -->
## Future Work

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
