![GitHub](https://img.shields.io/github/license/CascadingRadium/Air-Traffic-Distribution?style=flat)
[![GitHub forks](https://img.shields.io/github/forks/CascadingRadium/Air-Traffic-Distribution)](https://github.com/CascadingRadium/Air-Traffic-Distribution/network)
[![GitHub stars](https://img.shields.io/github/stars/CascadingRadium/Air-Traffic-Distribution)](https://github.com/CascadingRadium/Air-Traffic-Distribution/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/CascadingRadium/Air-Traffic-Distribution)](https://github.com/CascadingRadium/Air-Traffic-Distribution/issues)
![GitHub repo size](https://img.shields.io/github/repo-size/CascadingRadium/Air-Traffic-Distribution)
![GitHub last commit](https://img.shields.io/github/last-commit/CascadingRadium/Air-Traffic-Distribution)
<img src="https://developer.nvidia.com/favicon.ico" align ='right' width ='50'>
<h1> Air Traffic Distribution </h1>
<p align="justify">Capstone project done as a part of the requirements for the bachelors degree in CS at PES University (2019 - 23) guided by Dr. Preethi P.</p>
<p>
  <h3 align="center"> Contributors </h3> 
  <table align="center">
    <tr> 
      <th> Name </th>
      <th> SRN </th>
    </tr>
    <tr>
      <td> Rahul Rampure </td>
      <td> PES1UG19CS370 </td>
    </tr>
    <tr>
      <td> Raghav Tiruvallur </td>
      <td> PES1UG19CS362 </td>
    </tr>
    <tr>
      <td> Vybhav Acharya </td>
      <td> PES1UG19CS584 </td>
    </tr>
    <tr>
      <td> Shashank Navad </td>
      <td> PES1UG19CS601 </td>
    </tr>
    </table>
</p>
<h2 align="center">Abstract</h2>
<ul align="justify">
<li>The idea presented here is a Genetic Algorithm developed in CUDA and C that allows a flight dispatcher to input a flight schedule, which is a list of prospective flights with each flight’s departure and arrival airports, scheduled departure time and cruise speed.</li>
<li>The algorithm then generates paths for each of these flights so that the aeroplane encounters the least amount of mid-air traffic along its route, reducing air traffic density. We do so by considering the time-varying position of the plane and ensuring that the number of other aircraft near it remains minimal throughout its flight.</li>
<li>The algorithm considers adding a minimal delay to the departure of an aircraft as well since doing so would allow for a shorter route with lesser en-route traffic and compares the benefit of this method to a longer route which avoids most of the traffic.</li> 
<li>We add a constraint to the algorithm according to which the departure and arrival airports must have at least one runway available for the aeroplane to use at the time of departure and arrival, respectively, effectively performing an aircraft-to-runway mapping at the airports.</li>
<li>Hence, the output the dispatcher receives will be each flight’s actual departure time, which considers flight delays, and the optimal route for each aeroplane. We develop an interactive website that the dispatcher can use to enter/upload the schedule and execute the algorithm at the click of a button. Finally, we developed a python simulator which shows each aircraft’s position along its path over time.</li>
</ul>
<h2 align="center">System Requirements, Execution and Evaluation</h2>
<p align="justify"> The following set of requirements must be satisfied by the execution environment before the repository is cloned.</p>
<ul align="justify">
<li>Linux based OS (Windows/WSL not supported).</li>
<li>CUDA enabled GPU.<a href="https://developer.nvidia.com/cuda-gpus"> Check if your GPU is CUDA enabled.</a></li>
<li>NodeJS (Version > 18).<a href="https://nodejs.org/en/download/"> Download NodeJS.</a></li>
<li>Nvidia CUDA Compiler(NVCC).<a href="https://developer.nvidia.com/cuda-downloads?target_os=Linux"> Download NVCC.</a></li>
<li>Python (Version > 3.8).</li>
<li>Git and Git LFS  <code>sudo apt-get install git git-lfs</code> </li> 
</ul>
<p align="justify"> To execute the project, one needs to run the shell script “runWebsite.sh” provided in the root. The shell script will perform a first-time setup and launch the website. The user can input or upload a flight schedule, execute the algorithm to obtain the solution, and simulate the same using the buttons provided.</p>  

```
chmod +x runWebsite.sh
./runWebsite.sh
```

The video below shows the website in use:

https://user-images.githubusercontent.com/53022689/207081997-caba21ec-5c32-49b5-b5d3-af1a7a801299.mp4

The simulator is built using Python and the gifs presented below show the traffic being distributed for various test scenarios given as input.

<img src="https://user-images.githubusercontent.com/53022689/207095647-37317714-6522-4509-bda4-5aab50383b83.gif" width="250" height="250"/>
