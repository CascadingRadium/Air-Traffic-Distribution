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
let d=new Date();
d.setHours(23,0);
let startDate=addMinutes(d,60)
console.log(prettifyDate(startDate))
