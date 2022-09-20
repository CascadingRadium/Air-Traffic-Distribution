import './App.css';
import axios from 'axios'
import { useState , useEffect} from 'react';
import CustomizedTables from './components/AddTableRows';
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";
import {useNavigate} from 'react-router-dom';
import LoadingButton from './components/LoadingButton'

function App() {

  const [sourceAirports,setsourceAirports]=useState([])
  const [destinatonAirports,setdestinationAirports]=useState([])
  const [sourceAirport,setsourceAirport]=useState("")
  const [destinationAirport,setdestinationAirport]=useState("")
  const [numberOfFlights,setNumberOfFlights]=useState(1)
  const [startTime,setStartTime]=useState("")
  const [items,setItems]=useState([])
  const [states,setStates]=useState([])
  const [stateToAirport,setStateToAirport]=useState({})
  const [isLoading,setisLoading]=useState(false)
  const [file,setFile]=useState()
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
      const airports=Object.values(stateToAirportMapping)
      let airportList=[]
      airports.forEach((airport)=>{
        airportList = [...airportList, ...airport]
      })
      console.log(airportList , airportList.length)
      setsourceAirports(airportList)
      setdestinationAirports(airportList)
    })
  }


  const getPathHelper=async(e)=>{
    e.preventDefault();
    setisLoading(true)
    await axios.post("http://localhost:5000/api/get-paths",items)
    .then(({data})=>{
      console.log(data.data)
      setItems([])
    })
    setisLoading(false)
    navigate("/paths")

  }

  const uploadFile=(e)=>{
    e.preventDefault()
    const formData=new FormData()
    formData.append('file',file)
    console.log(formData)
    setFile()
    axios.post("http://localhost:5000/api/upload-file",formData)
    .then(({data})=>{
        setItems(items=>[...items,...data.data])
    })
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

  return (
    <>
    <div className="App">
      <h1>GPU - Accelerated Genetic Algorithm for Air Traffic Management</h1>
      <form>
      <label>
          Select State:
        <select onChange={(e)=>setsourceAirports(stateToAirport[e.target.value])}>
          <option value=""> Select a state </option>
        {
          states.map((state)=>(
            <option key={state} value={state}>{state}</option>
          ))
        }
        </select>
        </label>
        &emsp;
        &emsp;
        &emsp;
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
          <option value=""> Select a state </option>
        {
          states.map((state)=>(
            <option key={state} value={state}>{state}</option>
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
        <button type='submit' class="btn btn-primary" onClick={addFlight}>Add flight</button><br/><br/>
        <h3>---------------  OR ----------------</h3>
        &emsp;
        <input type='file' onChange={(e)=>setFile(e.target.files[0])}/>
        &emsp;
        <button type='submit' class="btn btn-primary" onClick={uploadFile}>Upload File</button>
      </form>
      
    </div>
    <div className='table'>
     
      <h1>Flights</h1>
      {
          <>
          <CustomizedTables items={items} deleteEntry={deleteEntry}/>

          {isLoading?<LoadingButton/>:<button className='btn btn-primary' onClick={getPathHelper}>Get Paths</button>}
          </>
          
      }
    </div>
      </>
 
  );
}

export default App;
