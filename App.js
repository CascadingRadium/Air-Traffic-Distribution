import './App.css';
import axios from 'axios'
import { useState , useEffect} from 'react';
import airportsData from './airports.txt'
import CustomizedTables from './components/AddTableRows';
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";
function App() {
  const [firstNumber,setfirstNumber]=useState(0);
  const [secondNumber,setsecondNumber]=useState(0);
  const [answer,setAnswer]=useState(0)
  const [answer1,setAnswer1]=useState(0)
  const [airports,setAirports]=useState([])
  const [sourceAirport,setsourceAirport]=useState("")
  const [destinationAirport,setdestinationAirport]=useState("")
  const [numberOfFlights,setNumberOfFlights]=useState(1)

  const [items,setItems]=useState([])
  const [simulatorData,setsimulatorData]=useState({})

  useEffect(()=>{
    getAirportData()
  },[])

  const getAirportData=()=>{
    fetch(airportsData)
  .then(r => r.text())
  .then(text => {
    let airportList=text.split("\n")
    let airportData=airportList.map(airport=>{
      const airportName=airport.split(",")[1]
      const airportState=airport.split(",")[2]
      return airportName + "," + airportState;
    });
    airportData=airportData.slice(0,airportData.length-1)
    //console.log(airports)
    airportData.sort()  
    setAirports(airportData)
  });
  }
  const calculateSum=(e)=>{
    e.preventDefault();
    const data={firstNumber,secondNumber}
    axios.post("http://localhost:5000/api/submit",data)
    .then(({data})=>{
      console.log(data.data)
      setAnswer(data.data)
    })

  }
  const calculateDifference=(e)=>{
    e.preventDefault();
    const data={firstNumber,secondNumber}
    axios.post("http://localhost:5000/api/submit-subtraction",data)
    .then(({data})=>{
      console.log(data.data)
      setAnswer1(data.data)
    })

  }

  const getPathHelper=(e)=>{
    e.preventDefault();
    axios.post("http://localhost:5000/api/get-paths",items)
    .then(({data})=>{
      console.log(data.data)
      setItems([])
    })

  }
  const addFlight=(e)=>{
    e.preventDefault();
    const sourceAirportName=sourceAirport.split(",")[0]
    const destinationAirportName=destinationAirport.split(",")[0]
    const data={sourceAirportName,destinationAirportName,numberOfFlights};
    setItems(items=>[...items,data])
  }

  const deleteEntry=(idx,e)=>{
      setItems(items.filter((v,i)=>i!==idx))
  }
  const mpld3_load_lib = (url, callback) => {
    var s = document.createElement('script');
    s.src = url;
    s.async = true;
    s.onreadystatechange = s.onload = callback;
    s.onerror = function () { console.warn("failed to load library " + url); };
    document.getElementsByTagName("head")[0].appendChild(s);
  }

  const continueSim=()=>{
    console.log("hello")
    axios.get("http://localhost:5000/api/simulation")
    .then(({data})=>{
      console.log(data.data)
    })
  }


  const fig_name="lol";
  return (
    <>
    <div className="App">
      <h1>Lol let's try</h1>
      <form>
        <label>
          Select Source Airport:
        <select onChange={(e)=>setsourceAirport(e.target.value)}>
        <option value=""> Select an airport </option>
        {
          airports.map((airport)=>(
            <option key={airport} value={airport}>{airport}</option>
          ))
        }
        </select>
        </label>
        &emsp;
        &emsp;
        &emsp;
        <label>
          Select Destination Airport:
        <select onChange={(e)=>setdestinationAirport(e.target.value)}>
        <option value=""> Select an airport </option>
        {
          airports.map((airport)=>(
            <option key={airport} value={airport}>{airport}</option>
          ))
        }
        </select>
        </label> 
        &emsp;

        Number of flights:
        <input type="number" value={numberOfFlights} onChange={(e)=> setNumberOfFlights(e.target.value)}/>
        <br/>
        <button type='submit' class="btn btn-primary" onClick={addFlight}>Add flight</button>
      </form>
      
    </div>
    <div className='table'>
     
      <h1>Flights</h1>
      {
          <>
          <CustomizedTables items={items} deleteEntry={deleteEntry}/>
          <button className='btn btn-primary' onClick={getPathHelper}>Get Paths</button>
          </>
          //mpld3.draw_figure("hello",simulation)
          
      }
    </div>
      </>
  //   <div onClick={continueSim}>
  //     {
  //     mpld3_load_lib("https://d3js.org/d3.v5.js", function () {
  //       mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.5.8.js", function () {
  //         mpld3.remove_figure(fig_name)
  //         mpld3.draw_figure(fig_name, simulation);
  //       })
  //     })
  //   }
  //   <div id={fig_name}></div>
  // </div>

  );

  


}

export default App;
