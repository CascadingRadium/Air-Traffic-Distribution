const fs=require('fs')

const sleep = ms => new Promise(r => setTimeout(r, ms));
const sleepFunction=async(ms)=>{
	let x=Date.now()
	await sleep(ms)
	let y=Date.now() - x
	console.log(y)
	fs.writeFileSync('timeTaken.txt',(y/1000).toString())
}
sleepFunction(1000)
