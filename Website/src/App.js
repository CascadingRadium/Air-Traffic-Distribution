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
	//const [startTime,setStartTime]=useState("")
	const [mins,setMin]=useState(0)
	const [hrs,sethrs]=useState(0)
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
				const stateToAirportMapping=data.data
				let stateList=Object.keys(stateToAirportMapping)
				stateList.sort()
				setStates(stateList)
				setStateToAirport(stateToAirportMapping)
				const airports=Object.values(stateToAirportMapping)
				let airportList=[]
				airports.forEach((airport)=>{
					airportList = [...airportList, ...airport]
				})
				setsourceAirports(airportList)
				setdestinationAirports(airportList)
			})
	}


	const getPathHelper=async(e)=>{
		e.preventDefault();
		if(items.length===0)
		{
			alert("Please enter the input flights")
			return ;
		}
		setisLoading(true)
		await axios.post("http://localhost:5000/api/get-paths",items)
			.then(({data})=>{
				setItems([])
			})
		setisLoading(false)
		navigate("/paths")

	}

	const uploadFile=(e)=>{
		e.preventDefault()
		const formData=new FormData()
		formData.append('file',file)
		setFile()
		axios.post("http://localhost:5000/api/upload-file",formData)
			.then(({data})=>{
				setItems(data.data)
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
		if(speed-'0'<=2 || speed==='')
		{
			alert("Flight Speed cannot be NULL and should be > 2")
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
			<h1>Air Traffic Management using a GPU - Accelerated Genetic Algorithm</h1>
			<br/><h3>Enter the flight schedule</h3>
			<br/>
			<form>
				<div id="MasterTwo">
					<div id="source">
						<p align="center"><h4>Departure Airport</h4></p>
						<p align="center"> State</p>
						<p align="center">
							<select onChange={(e)=>setsourceAirports(stateToAirport[e.target.value])}>
								<option value=""> Select a state </option>
									{
										states.map((state)=>(
											<option key={state} value={state}>{state}</option>
										))
									}
							</select>
						</p>
						<p align="center">Airport Name</p>
						<p align="center">
							<select onChange={(e)=>setsourceAirport(e.target.value)}>
								<option value=""> Select an airport </option>
									{
										sourceAirports.map((airport,idx)=>(
											<option key={idx} value={airport}>{airport}</option>
										))
									}
							</select>
						</p>
					</div>
					<div id="destination" align="right">
						<p align="center"><h4>Arrival Airport</h4></p>
						<p align="center">State</p>
						<p align="center">
							<select onChange={(e)=>setdestinationAirports(stateToAirport[e.target.value])}>
								<option value=""> Select a state </option>
								{
									states.map((state)=>(
										<option key={state} value={state}>{state}</option>
									))
								}
							</select>
						</p>
						<p align="center"> Airport Name </p>
						<p align="center">
							<select onChange={(e)=>setdestinationAirport(e.target.value)}>
								<option value=""> Select an airport </option>
								{
									destinatonAirports.map((airport,idx)=>(
										<option key={idx} value={airport}>{airport}</option>
									))
								}
							</select>
						</p>
					</div>
				</div>
				<div id="MasterThree">
					<div id="Speed">
					<br/>
						<p align="center">Cruise Speed in Knots</p>
						<p align="center">
							<input type="text" onChange={(e) => setSpeed(e.target.value)} />
						</p>
					</div>
					<div id="NumFlights">
						<br/>
						<p align="center">Number of flights</p>
						<p align="center">
							<select onChange={(e)=>setNumberOfFlights(e.target.value)}>
								{
									number_of_flights().map((airport)=>(
										<option key={airport} value={airport}>{airport}</option>
									))
								}
								</select>
						</p>
					</div>
					<div id="DepartureTime">
					<br/>
						<p align="center">Scheduled Departure Time</p>
						<p align="center">
							Hour&emsp;
							<select onChange={(e)=>sethrs(e.target.value)}>
							{
								gethrs().map((state)=>(
									<option key={state} value={state}>{state}</option>
								))
							}
							</select>
							&emsp;
							Minutes&emsp;
							<select onChange={(e)=>setMin(e.target.value)}>
								{
									getmins().map((state)=>(
										<option key={state} value={state}>{state}</option>
									))
								}
							</select>
						</p>
					</div>
				</div>
				<p align="center">
					<button type='submit' className="btn btn-primary" onClick={addFlight}>Add Flight</button>
				</p>
				<p align="center">
					<h3>---------------  OR ----------------</h3>
				</p>
				<p align="center">
					<input type='file' onChange={(e)=>setFile(e.target.files[0])}/>
					<button type='submit' className="btn btn-primary" onClick={uploadFile}>Upload File</button>
				</p>
			</form>
			</div>
			<div className='table'>
			{
				<>
				{
					isLoading?<LoadingButton/>:<button className='btn btn-primary' onClick={getPathHelper}>Execute the CUDA GA module to generate the solution</button>
				}
				<br/><br/>
				<CustomizedTables items={items} deleteEntry={deleteEntry}/>
				</>

			}
			</div>
		</>
	);
}
export default App;
