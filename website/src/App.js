import './App.css';
import axios from 'axios'
import { useState , useEffect} from 'react';
import airportsData from './airports.txt'
import CustomizedTables from './components/AddTableRows';
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";
import {useNavigate} from 'react-router-dom';


function App() {
  const [firstNumber,setfirstNumber]=useState(0);
  const [secondNumber,setsecondNumber]=useState(0);
  const [answer,setAnswer]=useState(0)
  const [answer1,setAnswer1]=useState(0)
  const [airports,setAirports]=useState([])
  const [sourceAirports,setsourceAirports]=useState([])
  const [destinatonAirports,setdestinationAirports]=useState([])
  const [sourceAirport,setsourceAirport]=useState("")
  const [destinationAirport,setdestinationAirport]=useState("")
  const [numberOfFlights,setNumberOfFlights]=useState(1)
  const [startTime,setStartTime]=useState("")
  const [items,setItems]=useState([])
  const [simulatorData,setsimulatorData]=useState({})
  const [pathData,setPathData]=useState([])
  const [states,setStates]=useState([])
  const [stateToAirport,setStateToAirport]=useState({})
  const navigate=useNavigate()

  useEffect(()=>{
    getStates()
  },[])

  const getStates=()=>{
    axios.get("http://localhost:5000/api/get-states")
    .then(({data})=>{
      //console.log(data.data)
      const stateToAirportMapping=data.data
      console.log(stateToAirportMapping)
      let stateList=Object.keys(stateToAirportMapping)
      stateList.sort()
      setStates(stateList)
      setStateToAirport(stateToAirportMapping)
    })
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

  const getPathHelper=async(e)=>{
    e.preventDefault();
    await axios.post("http://localhost:5000/api/get-paths",items)
    .then(({data})=>{
      console.log(data.data)
      setItems([])
    })
    navigate("/paths")

  }
  const addFlight=(e)=>{
    e.preventDefault();
    const sourceAirportName=sourceAirport.split(",")[0]
    const destinationAirportName=destinationAirport.split(",")[0]
    let dataList=[];
    const data={sourceAirportName,destinationAirportName,startTime};
    for(let i=0;i<numberOfFlights;i++)
      dataList.push(data)
    setItems(items=>[...items,...dataList])
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
          Select State:
        <select onChange={(e)=>setsourceAirports(stateToAirport[e.target.value])}>
        {
          states.map((state)=>(
            <option key={state} value={state}>{state}</option>
          ))
        }
        </select>
        </label>
        <label>
          Select Source Airport:
        <select onChange={(e)=>setsourceAirport(e.target.value)}>
        <option value=""> Select an airport </option>
        {
          sourceAirports.map((airport)=>(
            <option key={airport} value={airport}>{airport}</option>
          ))
        }
        </select>
        </label>
        &emsp;
        &emsp;
        &emsp;
        <label>
          Select State:
        <select onChange={(e)=>setdestinationAirports(stateToAirport[e.target.value])}>
        {
          states.map((state)=>(
            <option key={state} value={state}>{state}</option>
          ))
        }
        </select>
        </label>
        <label>
          Select Destination Airport:
        <select onChange={(e)=>setdestinationAirport(e.target.value)}>
        <option value=""> Select an airport </option>
        {
          destinatonAirports.map((airport)=>(
            <option key={airport} value={airport}>{airport}</option>
          ))
        }
        </select>
        </label> 
        &emsp;

        Number of flights:
        <input type="number" value={numberOfFlights} onChange={(e)=> setNumberOfFlights(e.target.value)}/>
        <br/>
        &emsp;
        Start Time:
        <input type="text" value={startTime} onChange={(e)=> setStartTime(e.target.value)}/>
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
          //<PathTable pathData={pathData}/>
          
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
