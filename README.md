![GitHub](https://img.shields.io/github/license/CascadingRadium/Air-Traffic-Distribution?style=flat)
[![GitHub forks](https://img.shields.io/github/forks/CascadingRadium/Air-Traffic-Distribution)](https://github.com/CascadingRadium/Air-Traffic-Distribution/network)
[![GitHub stars](https://img.shields.io/github/stars/CascadingRadium/Air-Traffic-Distribution)](https://github.com/CascadingRadium/Air-Traffic-Distribution/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/CascadingRadium/Air-Traffic-Distribution)](https://github.com/CascadingRadium/Air-Traffic-Distribution/issues)
![GitHub repo size](https://img.shields.io/github/repo-size/CascadingRadium/Air-Traffic-Distribution)
![GitHub last commit](https://img.shields.io/github/last-commit/CascadingRadium/Air-Traffic-Distribution)
<img src="https://developer.nvidia.com/favicon.ico" align ='right' width ='50'>
<h1> Air Traffic Distribution </h1>
Capstone project done as a part of the requirements for the bachelors degree in CS at PES University (2019 - 23) guided by Dr. Preethi P.
<p style="text-align:right">
  <p align="center">Contributors</p> <table align="center">
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
</p><br>
<p>
The idea that is being presented here is a Genetic Algorithm developed in CUDA and C that allows a flight dispatcher to input a flight schedule, which is a list of prospective flights with each flight's departure and arrival airports, scheduled departure time and cruise speed. The algorithm then generates paths for each of these flights in such a way that the airplane encounters the least amount of mid-air traffic along its route, leading to a reduction in the air-traffic density. We do so by considering the time-varying position of the plane and ensuring that the number of other aircraft near it remains minimal throughout its flight. The algorithm considers adding a minimal delay to the departure of an aircraft as well, since doing so would allow for a shorter route with lesser enroute traffic and compares the benefit of this method to a longer route which avoids most of the traffic. We add a constraint to the algorithm according to which the departure and arrival airports must have atleast one runway available for the airplane to use at the time of departure and arrival respectively. Hence, the output the dispatcher recieves will be each flight's actual departure time, which takes delays into consideration, and the optimal route for each airplane. We develop an interactive website that the dispatcher can use to enter/upload the schedule and execute the algorithm on the click of a button. Finally we develop a simulator in python which shows the aposition of each aircraft along its path with respect to time.
</p>
