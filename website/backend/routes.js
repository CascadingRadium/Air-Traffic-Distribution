const router = require('express').Router();
const execSync = require('child_process').execSync;
const fs = require('fs');
const airportsData = '../src/airports.txt'
const paths='./example.txt'

var airportSectorMapping={}


let flightIDToAirportMapping={}


const data = fs.readFileSync(airportsData).toString().split("\n");

const pathsFile=fs.readFileSync(paths).toString().split("\n")


const prettifyDate=(date)=>{

  const hours=date.getHours()
  const minutes=date.getMinutes();
  let hourString,minuteString=minutes.toString();
  let subString="AM";
  if(minutes < 10)
  {
    minuteString="0" + minuteString;
  }
  if(hours >=12)
  {
    if(hours!==12)
    {
      hours-=12
    }
    subString="PM";
  }
  hourString=hours.toString();
  return `${hourString}:${minuteString} ${subString}` 

}






function addMinutes(date, minutes) {
  return new Date(date.getTime() + minutes*60000);
}


data.map((airportInfo)=>{
  const airportData=airportInfo.split(",")
  airportSectorMapping[airportData[1]]=airportData[3]
})


router.post("/submit",(req,res)=>{
    console.log(req.body)
    console.log(`./a.out ${req.body.firstNumber} ${req.body.secondNumber}`)
    execSync('./a.out ' + req.body.firstNumber +' '+ req.body.secondNumber, { encoding: 'utf-8' }); 
    try {
      const data = fs.readFileSync('output.txt', 'utf8');
      res.send({"data":data})
      } catch (err) {
        console.error(err);
      }
    })
    
router.post("/submit-subtraction",(req,res)=>{
    console.log(req.body)
    console.log(`python3 ${req.body.firstNumber} ${req.body.secondNumber}`)
    execSync('python3 subtract.py ' + req.body.firstNumber +' '+ req.body.secondNumber, { encoding: 'utf-8' }); 
    try {
      const data = fs.readFileSync('output-subtract.txt', 'utf8');
        res.send({"data":data})
      } catch (err) {
        console.error(err);
      }
})

router.post("/get-paths",async(req,res)=>{
    

    const flights=req.body
    let content="";
    flights.map((flight,id)=>{
      const startTime=flight.startTime
      const sourceAirport=flight.sourceAirportName
      const destinationAirport=flight.destinationAirportName
      const sourceSector=airportSectorMapping[sourceAirport]
      const destinationSector=airportSectorMapping[destinationAirport]
      flightIDToAirportMapping[id]={startTime,sourceAirport,destinationAirport}
      console.log(flightIDToAirportMapping)
      content+=`${sourceSector},${destinationSector}\n`
    })
  try {
    fs.writeFileSync('od_pairs.txt', content);
    res.status(200).json({"data":"Paths generated"})
  } catch (err) {
    console.error(err);
    res.status(500).json({"data":err})
  }

})

router.get("/simulator",(req,res)=>{

  try{
  execSync('ipython -c "%run Simulator.ipynb"', { encoding: 'utf-8' });
  return res.status(200).json({"data":"Success"})
  }
  catch(e)
  {
    return res.status(500).json({"data":"Something is wrong"})
  }
})

router.get("/get-times",(req,res)=>{

  try{
  timeObj={}
  let timeList=[]
  pathsFile.forEach((path,id)=>{
    const start=flightIDToAirportMapping[id].startTime.split(":")
    let d=new Date();
    d.setHours(start[0],start[1]);
    const pathData=path.split(" ")
    const startTime=5*parseInt(pathData[1]) 
    const endTime=5*parseInt(pathData[2])
    let startDate=addMinutes(d,startTime)
    let endDate=addMinutes(d,endTime)
    let sourceAirport=flightIDToAirportMapping[id].sourceAirport
    let destinationAirport=flightIDToAirportMapping[id].destinationAirport
    startDate=prettifyDate(startDate)
    endDate=prettifyDate(endDate)
    timeObj={ id,sourceAirport,destinationAirport,startDate,endDate}
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