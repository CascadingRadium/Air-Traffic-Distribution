
import App from './App';
import React from 'react';
import {BrowserRouter, Route, Routes} from 'react-router-dom';
import PathTable from './components/PathsComponent';
import './App.css';
import Simulator from './components/Simulator';
function Application(){
    return(
    <BrowserRouter>
        <Routes>
            <Route path="/" element={<App/>} />
            <Route path="/paths" element={<PathTable/>} />
            <Route path='/sim' element={<Simulator/>}/>
        </Routes>
      </BrowserRouter>
    );
}

export default Application;