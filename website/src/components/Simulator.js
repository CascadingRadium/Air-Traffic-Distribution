import * as React from 'react';
import {useState} from 'react';
import './Style.css'
import background from '../my_plot.png'
function Simulator(){

    const [image,setImage]=useState("../my_plot0.png")
    return(
        <>
        <canvas id="canvas" width="2000" height="1500" />
        </>
    )
    }
export default Simulator;