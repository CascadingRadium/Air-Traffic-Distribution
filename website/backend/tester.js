let d=new Date();
d.setHours(0,0);
function addMinutes(date, minutes) {
	return new Date(date.getTime() + minutes*60000);
}
const prettifyDate=(date)=>{

	let hours=date.getHours()
	const minutes=date.getMinutes();
	let hourString=hours.toString(),minuteString=minutes.toString();
	if(minutes < 10)
		minuteString="0" + minuteString
	return `${hourString}:${minuteString}` 

}
let startDate=addMinutes(d,1037)
let endDate=addMinutes(d,1080)
startDate=prettifyDate(startDate)
endDate=prettifyDate(endDate)
console.log(startDate,endDate)