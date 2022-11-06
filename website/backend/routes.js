const router = require('express').Router();
const execSync = require('child_process').execSync;
const fs = require('fs');
const airportsData = '../src/AirportFileForFrontend.txt'
const paths='./OutputToFrontend.txt'
const CONVERSION_FACTOR=1000*60;
var airportSectorMapping={}


var flightIDToAirportMapping={}
var stateToAirports={}
var airportICAOMapping={}
var airportToXYMapping={}
const data = fs.readFileSync(airportsData).toString().split("\n");

const prettifyDate=(date)=>{

	let hours=date.getHours()
	const minutes=date.getMinutes();
	let hourString=hours.toString(),minuteString=minutes.toString();
	if(minutes < 10)
		minuteString="0" + minuteString
	return `${hourString}:${minuteString}` 

}

function addMinutes(date, minutes) {
	return new Date(date.getTime() + minutes*60000);
}


data.map((airportInfo)=>{
	const airportData=airportInfo.split(",")
	airportSectorMapping[airportData[0]]=airportData[2]
	airportICAOMapping[airportData[0]]=airportData[3]
	airportToXYMapping[airportData[0]]=[airportData[4] , airportData[5]]
})


router.post("/get-paths",async(req,res)=>{


	const flights=req.body
	let content="";
	flights.map((flight,id)=>{
		const startTime=flight.startTime.split(":")
		const difference=(Number(startTime[0]))*60 + (Number(startTime[1])) + 60
		const idx=difference
		const sourceAirport=flight.sourceAirportName
		const destinationAirport=flight.destinationAirportName
		const sourceSector=airportSectorMapping[sourceAirport]
		const destinationSector=airportSectorMapping[destinationAirport]
		const sourceICAOCode=airportICAOMapping[sourceAirport]
		const destinationICAOCode=airportICAOMapping[destinationAirport]
		const sourceAirportCoordinates=airportToXYMapping[sourceAirport]
		const destinationAirportCoordinates=airportToXYMapping[destinationAirport]
		const speed=flight.speed
		flightIDToAirportMapping[id]={startTime,sourceAirport,destinationAirport,speed}
		content+=`${sourceSector} ${sourceICAOCode} ${sourceAirportCoordinates[0]} ${sourceAirportCoordinates[1]},${destinationSector} ${destinationICAOCode} ${destinationAirportCoordinates[0]} ${destinationAirportCoordinates[1]},${idx},${speed}\n`
	})
	try {
		fs.writeFileSync('InputFromFrontend.txt', content);
		let start=Date.now()
		execSync('./a.out')
		let timeTaken= Date.now() - start 
		console.log(timeTaken / 1000)
		fs.writeFileSync('timeTaken.txt',(timeTaken/1000).toString())
		res.status(200).json({"data":"Paths generated"})
	} catch (err) {
		console.error(err);
		res.status(500).json({"data":err})
	}

})

router.get("/simulator/:speed",(req,res)=>{

	try{
		const speed=req.params.speed;
		execSync(`python3 sim.py ${speed}`, { encoding: 'utf-8' });
		res.status(200).json({"data":"Success"})
	}
	catch(e)
	{
		res.status(500).json({"data":"Something is wrong"})
	}
})

router.post("/upload-file",(req,res)=>{
	const data=req.files.file.data.toString().split("\n")
	if(data[data.length -1]=='')
	{
		data.pop()
	}
	let items=[]
	data.forEach((row)=>{
		const info=row.split(",")
		const sourceAirportName=info[0]
		const destinationAirportName=info[1]
		const numberOfFlights=parseInt(info[2])
		const startTime=info[3]
		const speed=info[4]
		const item={startTime,sourceAirportName,destinationAirportName,speed}
		for(let i=0;i<numberOfFlights;i++)
		{
			items.push(item)
		}
	})
	res.status(200).json({"data":items})
})

router.get("/get-states",(req,res)=>{

	try{
		data.forEach((airportData)=>{
			const airportName=airportData.split(",")[0]
			const state=airportData.split(",")[1]
			if(stateToAirports[state]===undefined)
				stateToAirports[state]=[airportName]
			else if(!stateToAirports[state].includes(airportName))
				stateToAirports[state].push(airportName)
		})
		res.status(200).json({"data":stateToAirports})
	}
	catch(e)
	{
		res.status(500).json({"data":e})
	}

})

router.get("/get-times",async(req,res)=>{

	try{
		timeObj={}
		let timeList=[]
		const pathsFile=fs.readFileSync(paths).toString().split("\n")
		pathsFile.forEach((path,id)=>{
			const start=flightIDToAirportMapping[id].startTime
			let d=new Date();
			d.setHours(0,0);
			const pathData=path.split(",")
			const startTime=parseInt(pathData[pathData.length -6]) 
			const endTime=parseInt(pathData[pathData.length - 4])
			let startDate=addMinutes(d,startTime)
			let endDate=addMinutes(d,endTime)
			let sourceAirport=flightIDToAirportMapping[id].sourceAirport
			let destinationAirport=flightIDToAirportMapping[id].destinationAirport
			const aerialTime=pathData[pathData.length - 5]
			const groundHolding=pathData[pathData.length - 7]
			startDate=prettifyDate(startDate)
			endDate=prettifyDate(endDate)
			timeObj={id,sourceAirport,destinationAirport,startDate,endDate,aerialTime,groundHolding}
			timeList.push(timeObj)
		})
		res.status(200).json({"data":timeList}) 
	}
	catch(e)
	{
		res.status(500).json({"data":e})  
	}

})
module.exports=router;
