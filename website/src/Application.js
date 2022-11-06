import App from './App';
import React from 'react';
import {BrowserRouter, Route, Routes} from 'react-router-dom';
import PathTable from './components/PathsComponent';
import './App.css';
function Application(){
    return(
    <BrowserRouter>
        <Routes>
            <Route path="/" element={<App/>} />
            <Route path="/paths" element={<PathTable/>} />
        </Routes>
      </BrowserRouter>
    );
}

export default Application;
