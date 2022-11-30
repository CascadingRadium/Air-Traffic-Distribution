import * as React from 'react';
import { styled } from '@mui/material/styles';
import Table from '../../node_modules/@mui/material/Table'
import TableBody from '../../node_modules/@mui/material/TableBody'
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '../../node_modules/@mui/material/Paper'
import '../App.css'
import { useEffect ,useState} from 'react';
import axios from 'axios';
import { useGlobalState } from '../states';

const StyledTableCell = styled(TableCell)(({ theme }) => ({
	[`&.${tableCellClasses.head}`]: {
		backgroundColor: theme.palette.common.black,
		color: theme.palette.common.white,
	},
	[`&.${tableCellClasses.body}`]: {
		fontSize: 14,
	},
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
	'&:nth-of-type(odd)': {
		backgroundColor: theme.palette.action.hover,
	},
	'&:last-child td, &:last-child th': {
		border: 0,
	},
}));

function createData(id,sourceAirport,destinationAirport,startDate,endDate,aerialTime,groundHolding) {
	return {id,sourceAirport,destinationAirport,startDate,endDate,aerialTime,groundHolding}
}


export default function PathTable() {
	
	
	const [pathData,setPathData]=useState([])
	const [value]=useGlobalState("value")
	useEffect(()=>{
		getPathData()
	},[])

	const goToSim=()=>{
		axios.get(`http://localhost:5000/api/simulator/${value}`)
			.then(({data})=>{
			})
	}


	const getPathData=()=>{
		axios.get("http://localhost:5000/api/get-times")
			.then(({data})=>{
				setPathData(data.data)
			})
	}
	const rows=pathData.map((item)=>{
		return createData(item.id,item.sourceAirport,item.destinationAirport,item.startDate,item.endDate,item.aerialTime,item.groundHolding)
	});
	return (
		<>
		<h1 align="center">Solution Generated</h1>
		<TableContainer component={Paper}>
		<Table sx={{ minWidth: 900 }} aria-label="customized table">
		<TableHead>
		<TableRow>
		<StyledTableCell><span className="TableCenterAlign">Flight ID</span></StyledTableCell>
		<StyledTableCell><span className="TableCenterAlign">Departure Airport</span></StyledTableCell>
		<StyledTableCell><span className="TableCenterAlign">Arrival Airport</span></StyledTableCell>
		<StyledTableCell><span className="TableCenterAlign">Actual Departure Time</span></StyledTableCell>
		<StyledTableCell><span className="TableCenterAlign">Arrival Time</span></StyledTableCell>
		<StyledTableCell><span className="TableCenterAlign">Aerial Time in minutes</span></StyledTableCell>
		<StyledTableCell><span className="TableCenterAlign">Flight Delay in minutes</span></StyledTableCell>
		</TableRow>
		</TableHead>
		<TableBody>
		{
			rows.map((row,idx) => (
				<StyledTableRow key={idx}>
				<StyledTableCell><span className="TableCenterAlign">{idx}</span></StyledTableCell>
				<StyledTableCell><span className="TableCenterAlign">{row.sourceAirport}</span></StyledTableCell>
				<StyledTableCell><span className="TableCenterAlign">{row.destinationAirport}</span></StyledTableCell>
				<StyledTableCell><span className="TableCenterAlign">{row.startDate}</span></StyledTableCell>
				<StyledTableCell><span className="TableCenterAlign">{row.endDate}</span></StyledTableCell>
				<StyledTableCell><span className="TableCenterAlign">{row.aerialTime}</span></StyledTableCell>
				<StyledTableCell><span className="TableCenterAlign">{row.groundHolding}</span></StyledTableCell>
				</StyledTableRow>
			))
		}
		</TableBody>
		</Table>
		</TableContainer>
		<br/>
		<p align="center">
			<button type='submit' className="btn btn-primary" onClick={goToSim}>Run the Simulator</button><br/>
		</p>
		</>

	);
}
