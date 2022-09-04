const express = require('express');
const PORT=5000
const app=express();
const cors = require('cors')
const apiRoutes = require("./routes")
const mongoose = require('mongoose');

app.use(express.json());
app.use(cors());
app.use("/api",apiRoutes)

const {MongoClient}=require('mongodb');
const uri="mongodb+srv://raghav-tiruvallur:qwertyDUDE@cluster0.1npdfrx.mongodb.net/flights";

mongoose.connect(uri,{useNewUrlParser:true,useUnifiedTopology:true},()=>{
    console.log("mongodb connected")
})




app.listen(PORT,() => {
    console.log(`Server is running on port ${PORT}`);
})