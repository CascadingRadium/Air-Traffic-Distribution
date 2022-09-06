import * as React from 'react';
import { styled } from '@mui/material/styles';
import Table from '../../node_modules/@mui/material/Table'
import TableBody from '../../node_modules/@mui/material/TableBody'
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '../../node_modules/@mui/material/Paper'
import { MdDelete } from 'react-icons/md' 
import '../App.css'
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

function createData(id,source,destination,startTime) {
  return { id,source,destination,startTime};
}


export default function CustomizedTables({items,deleteEntry}) {
  
    const rows=items.map((item)=>{
       return createData(item._id,item.sourceAirportName,item.destinationAirportName,item.startTime)
    });

   

    return (
        <>
    <TableContainer component={Paper}>
      <Table sx={{ minWidth: 700 }} aria-label="customized table">
        <TableHead>
          <TableRow>
          <StyledTableCell>Flight ID</StyledTableCell>
            <StyledTableCell>Source Airport</StyledTableCell>
            <StyledTableCell>Destination Airport</StyledTableCell>
          </TableRow>
        </TableHead>
        <TableBody>
        {
          rows.map((row,idx) => (
            <StyledTableRow key={idx}>
              <StyledTableCell>{idx}</StyledTableCell>
              <StyledTableCell>{row.source}</StyledTableCell>
              <StyledTableCell>{row.destination}</StyledTableCell>
            </StyledTableRow>
          ))
        }
        </TableBody>
      </Table>
    </TableContainer>
    
    </>
    
  );
}
