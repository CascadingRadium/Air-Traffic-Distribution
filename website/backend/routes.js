const router = require('express').Router();
const execSync = require('child_process').execSync;
const fs = require('fs');
const Flight = require('./Models/flightSchema');
const airportsData = '../some-trial/src/airports.txt'


var airportSectorMapping={}

const data = fs.readFileSync(airportsData).toString().split("\n");

data.map((airportInfo)=>{
   const airportData=airportInfo.split(",")
   airportSectorMapping[airportData[1]]=airportData[3]
})

console.log(airportSectorMapping)

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
    flights.map((flight)=>{
      const sourceAirport=flight.sourceAirportName
      const destinationAirport=flight.destinationAirportName
      const numberOfFlights=flight.numberOfFlights
      const sourceSector=airportSectorMapping[sourceAirport]
      const destinationSector=airportSectorMapping[destinationAirport]
      content+=`${sourceSector},${destinationSector}\n`.repeat(numberOfFlights)
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


module.exports=router;