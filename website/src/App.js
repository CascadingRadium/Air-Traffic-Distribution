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
	const [mins,setMin]=useState(0)
	const [hrs,sethrs]=useState(1)
	const [items,setItems]=useState([])
	const [states,setStates]=useState([])
	const [stateToAirport,setStateToAirport]=useState({})
	const [isLoading,setisLoading]=useState(false)
	const [file,setFile]=useState()
	const [speed,setSpeed]=useState("")
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
	const number_of_flights=()=>{
		let ar=[]
		for(let i=1;i<=20;i++){ar.push(i);}
		return ar
	}



	const gethrs=()=>{
		let arr=[]
		for(let i=0;i<24;i++)
		{
			arr.push(i)
		}
		return arr;
	}

	const getmins=()=>{
		let arr=[]
		for(let i=0;i<=59;i++)
		{
			arr.push(i)
		}
		return arr
	}
	const addFlight=(e)=>{
		e.preventDefault();
		const sourceAirportName=sourceAirport.split(",")[0]
		const destinationAirportName=destinationAirport.split(",")[0]
		if(sourceAirportName===destinationAirportName)
		{
			alert("Source airport and destination airport cannot be same")
			return ;
		}
		let dataList=[];
		let startTime=hrs.toString() + ":"
		if(mins < 10)
		{
			startTime+="0"+mins.toString()
		}
		else 
		{
			startTime +=mins.toString()
		}
		const data={sourceAirportName,destinationAirportName,startTime,speed};
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
		<h1>Flight Scheduler and Plan Generator</h1>
		<form>
		<div id="source" align="left">
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
		<br/>
		<br/>
		<br/>
		<label>
		Select Source Airport:
		<select onChange={(e)=>setsourceAirport(e.target.value)}>
		<option value=""> Select an airport </option>
		{
			sourceAirports.map((airport,idx)=>(
				<option key={idx} value={airport}>{airport}</option>
			))
		}
		</select>
		</label>
		</div>
		&emsp;
		&emsp;
		&emsp;
		<div id="destination" align="right">
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
		<br/>
		<br/>
		<br/>
		<label>
		Select Destination Airport:
		<select onChange={(e)=>setdestinationAirport(e.target.value)}>
		<option value=""> Select an airport </option>
		{
			destinatonAirports.map((airport,idx)=>(
				<option key={idx} value={airport}>{airport}</option>
			))
		}
		</select>
		</label> 
		</div>
		<br/>

		<label>
		Number of flights 
		&emsp;
	<select onChange={(e)=>setNumberOfFlights(e.target.value)}>
		{
			number_of_flights().map((airport)=>(
				<option key={airport} value={airport}>{airport}</option>
			))
		}
		</select>
		</label>
		<br/>
		&emsp;
		Speed:
		<input type="text" onChange={(e) => setSpeed(e.target.value)} /> Knots
		&emsp;
	<label>Hour: &emsp;
	<select onChange={(e)=>sethrs(e.target.value)}>
		{
			gethrs().map((state)=>(
				<option key={state} value={state}>{state}</option>
			))
		}
		</select>
		</label>
		&emsp;
	<label>Minutes: &emsp;
	<select onChange={(e)=>setMin(e.target.value)}>
		{
			getmins().map((state)=>(
				<option key={state} value={state}>{state}</option>
			))
		}
		</select>
		</label>
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
