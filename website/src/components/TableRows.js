function TableRows({airports,rowsData, deleteTableRows, handleChange}) {

    return(
        
        rowsData.map((data, index)=>{
            const {sourceAirport,destinationAirport,numberOfFlights}= data;
            return(

                <tr key={index}>
                <td>
                 {
                    sourceAirport
                 }
                </td>
                <td>
                {
                    destinationAirport
                }
                </td>
                <td>
                {
                    numberOfFlights
                }
                </td>
            </tr>

            )
        })
   
    )
    
}

export default TableRows;